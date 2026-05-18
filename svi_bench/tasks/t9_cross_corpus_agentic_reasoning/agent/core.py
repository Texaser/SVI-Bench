from typing import List, Dict, Union, Optional, Tuple, Iterator, Any
import sys
import os
import json
import time
import random
import json5
import re
import logging
from datetime import datetime

from qwen_agent.agents import FnCallAgent
from qwen_agent.tools import BaseTool
from qwen_agent.llm.schema import Message

logger = logging.getLogger(__name__)

# Defensive: when the agent intends to give up, the prompt instructs it to
# emit `<answer>I cannot find the answer in the provided documents or videos.</answer>`.
# If the model produces the verbatim phrase but forgets the <answer> tags,
# treat the bare phrase as a valid termination too (otherwise the parser
# injects an Error, the agent restarts the search, and we waste turns).
NO_INFO_FALLBACK_PHRASE = "I cannot find the answer in the provided documents or videos."


def _is_termination(content: str) -> bool:
    """Return True if `content` should terminate the agent loop.

    Primary signal is the `<answer>` tag (per system prompt). The bare
    fallback phrase is also accepted to absorb the case where the LLM
    forgot to wrap it in tags.
    """
    return ("<answer>" in content) or (NO_INFO_FALLBACK_PHRASE in content)


def robust_extract_tool_calls(content: str) -> List[Dict]:
    """
    Extract tool calls from LLM output content with high resilience.
    Handles:
    - <tool_call>{...}</tool_call> (standard)
    - <tool_call>{...} (unclosed)
    - {...} (unopened/naked JSON)
    - <think> leakage inside tool calls
    - Malformed JSON using json5
    """
    extracted_calls = []
    
    # 1. First, try to find all <tool_call> tags (closed or open)
    # Regex for closed tags
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    
    # If no closed tags, check for an unclosed tag at the end
    if not matches:
        unclosed_match = re.search(r'<tool_call>(.*)', content, re.DOTALL)
        if unclosed_match:
            matches = [unclosed_match.group(1).strip()]

    # 2. If STILL no tags found, try to see if there's a closing tag only or raw JSON
    if not matches:
        # 2a. Check for (possible text) + JSON + </tool_call>
        closing_only = re.findall(r'(.*)</tool_call>', content, re.DOTALL)
        if closing_only:
             # Try to find the start of JSON inside the captured part
             for potential in closing_only:
                 json_start = potential.find('{')
                 if json_start != -1:
                     matches.append(potential[json_start:].strip())
        
        # 2b. If still nothing, look for naked JSON object containing "name" and "arguments"
        if not matches:
             # Look for JSON-like objects starting with {"name":
             for m in re.finditer(r'\{[\s\n]*"name":', content):
                 start_idx = m.start()
                 matches.append(content[start_idx:].strip())

    # 3. Check for [Tool Call]: name(args) pattern
    if not matches:
        func_call_match = re.search(r'\[Tool Call\]:\s*([a-zA-Z0-9_]+)\((.*)\)', content, re.DOTALL)
        if func_call_match:
             name = func_call_match.group(1)
             args_str = func_call_match.group(2).strip()
             args = {}
             if args_str:
                 try:
                     # Try to parse arguments. Supports JSON args inside parens
                     args = json5.loads(args_str)
                 except Exception:
                     # If parsing fails, pass empty args to trigger tool validation error later
                     pass
             extracted_calls.append({'name': name, 'arguments': args})
             return extracted_calls

    # 3. Process matches with json5
    for raw_call in matches:
        # Clean up thinking block leakage if it somehow got inside
        cleaned_call = re.sub(r'<think>.*?</think>', '', raw_call, flags=re.DOTALL).strip()
        cleaned_call = re.sub(r'<think>.*', '', cleaned_call, flags=re.DOTALL).strip()
        
        if not cleaned_call:
            continue

        # Try to find a valid JSON object by balancing braces starting from the first '{'
        json_start = cleaned_call.find('{')
        if json_start != -1:
            potential_json = cleaned_call[json_start:]
            balance = 0
            json_end = -1
            for i, char in enumerate(potential_json):
                if char == '{': balance += 1
                elif char == '}': balance -= 1
                if balance == 0:
                    json_end = i + 1
                    break
            
            if json_end != -1:
                final_json_str = potential_json[:json_end]
                try:
                    data = json5.loads(final_json_str)
                    if isinstance(data, dict) and 'name' in data:
                        extracted_calls.append(data)
                        continue # Found it!
                except (ValueError, TypeError):
                    pass

            # Conservative repair: if the brace counter ended with `balance > 0`,
            # the model dropped that many closing braces (a known MiniMax-M2.5
            # drift; harmless for Qwen/GPT which produce valid JSON in the first
            # branch above). Append the missing closers and re-try.
            # Strict validation: only accept the repair if it produces an object
            # with a string `name` AND a dict `arguments` — won't invent fields.
            if json_end == -1 and balance > 0:
                repaired = potential_json + ('}' * balance)
                try:
                    data = json5.loads(repaired)
                    if (
                        isinstance(data, dict)
                        and isinstance(data.get('name'), str)
                        and isinstance(data.get('arguments'), dict)
                    ):
                        logger.warning(
                            "tool_call JSON repair: appended %d closing brace(s). "
                            "name=%r. Raw snippet: %r",
                            balance, data['name'], raw_call[:200],
                        )
                        extracted_calls.append(data)
                        continue
                except Exception:
                    pass

        # Fallback to loading the whole thing if balancing failed
        try:
            data = json5.loads(cleaned_call)
            if isinstance(data, dict) and 'name' in data:
                extracted_calls.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        extracted_calls.append(item)
        except Exception:
            continue
            
    return extracted_calls

# Imports for GPT/Manual Agent
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError


def qwen_tool_to_openai_tool(tool: BaseTool) -> Dict:
    """
    Convert a Qwen-Agent BaseTool definition to OpenAI 'tools' format.
    """
    properties = {}
    required_params = []
    
    for param in tool.parameters:
        param_name = param['name']
        param_type = param['type']
        param_desc = param.get('description', '')
        
        # Basic type mapping
        if param_type == 'array':
            # Ensure items is present if type is array
            items_config = param.get('items', {'type': 'string'})
            prop_def = {
                "type": "array",
                "description": param_desc,
                "items": items_config
            }
        else:
            prop_def = {
                "type": param_type,
                "description": param_desc
            }
            
        properties[param_name] = prop_def
        
        if param.get('required', False):
            required_params.append(param_name)
            
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        }
    }


def generate_tool_prompt_xml(function_map: Dict) -> str:
    """
    Generate XML-based tool instructions for custom agents.
    Returns a string to be appended to the system prompt.
    """
    if not function_map:
        return ""
    
    tool_defs = []
    for tool in function_map.values():
        tool_def = qwen_tool_to_openai_tool(tool)
        tool_defs.append(json.dumps(tool_def))
    
    tools_xml = "\n".join(tool_defs)
    
    prompt = f"""

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_xml}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
    
    return prompt
    


# ==============================================================================
# Custom Agent Classes - Unified XML-based Tool Calling
# ==============================================================================

class CustomQwenMultiTurnReactAgent(FnCallAgent):
    """
    Custom Qwen Agent using custom call_server() and XML-based tool parsing.
    Does not rely on qwen-agent's built-in LLM calling.
    """
    def __init__(self,
                 llm_config: Dict,
                 system_message: str,
                 function_list: List[BaseTool],
                 verbosity_level: int = 1,
                 max_llm_calls: int = 20,
                 max_retries: int = 5,
                 retry_base_sleep: int = 2,
                 max_context_tokens: int = 110000,
                 pruning_token_limit: int = 100000,
                 prompts_dir: str = None,
                 **kwargs):
        
        self.verbosity = verbosity_level
        self.model_name = llm_config.get('model', 'unknown')
        self.model_server = llm_config.get('model_server', None)
        self.api_key = llm_config.get('api_key', "EMPTY")
        self.generate_cfg = llm_config.get('generate_cfg', {})
        self.max_llm_calls = max_llm_calls
        self.max_retries = max_retries
        self.retry_base_sleep = retry_base_sleep
        self.max_context_tokens = max_context_tokens
        self.pruning_token_limit = pruning_token_limit

        self.system_message = system_message
        self.name = kwargs.get('name')
        self.description = kwargs.get('description')

        self.function_map = {}
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

        self.tool_prompt = generate_tool_prompt_xml(self.function_map)
        self.client = OpenAI(api_key=self.api_key, base_url=self.model_server)

        from agent.context_manager import ContextManager
        self.context_manager = ContextManager(token_limit=self.pruning_token_limit, max_context_tokens=self.max_context_tokens, model_name=self.model_name, prompts_dir=prompts_dir)

    def run(self, messages: List[Union[Dict, Message]], **kwargs) -> Iterator[List[Message]]:
        """Simplified run - passes messages directly to _run."""
        from qwen_agent.llm.schema import Message
        import copy
        
        # make the message is properply formated in Message.
        messages = copy.deepcopy(messages)
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
        
        # _run runs while and yield the response.
        for rsp in self._run(messages=new_messages, **kwargs):
            yield rsp

    def call_server(self, msgs: List[Dict], max_tries: int = 10) -> str:
        """Custom LLM server call with retry logic."""
        base_sleep_time = self.retry_base_sleep
        
        for attempt in range(max_tries):
            try:

                chat_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.generate_cfg.get('temperature', 0.6),
                    top_p=self.generate_cfg.get('top_p', 0.95),
                    max_tokens=self.generate_cfg.get('max_tokens', 10000)
                )

                content = chat_response.choices[0].message.content
                if content and content.strip():
                    # Extract usage
                    usage = getattr(chat_response, 'usage', None)
                    usage_dict = {}
                    if usage:
                        try:
                            usage_dict = usage.model_dump()
                        except (AttributeError, TypeError):
                            try:
                                usage_dict = dict(usage)
                            except (TypeError, AttributeError, ValueError):
                                pass
                        
                        # Ensure standard keys exist
                        usage_dict.setdefault('prompt_tokens', getattr(usage, 'prompt_tokens', 0))
                        usage_dict.setdefault('completion_tokens', getattr(usage, 'completion_tokens', 0))
                        usage_dict.setdefault('total_tokens', getattr(usage, 'total_tokens', 0))
                    return content.strip(), usage_dict
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")
                    
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")
            
            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return "Error: LLM server call failed after all retries.", {}

    def custom_call_tool_with_raw(self, tool_name: str, tool_args: Dict) -> Tuple[str, Any]:
        """Execute tool and return both string output and raw result."""
        if tool_name in self.function_map:
            try:
                raw_result = self.function_map[tool_name].call(tool_args)
                return str(raw_result), raw_result
            except Exception as e:
                err = f"Error executing {tool_name}: {e}"
                return err, None
        else:
            return f"Error: Tool {tool_name} not found", None

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Custom Qwen ReAct loop with raw tool output capture."""
        from qwen_agent.llm.schema import FUNCTION, Message
        import copy
        
        # Prepare initial messages for Context Manager with SYSTEM PROMPT
        system_content = self.system_message + generate_tool_prompt_xml(self.function_map)
        initial_context = [{"role": "system", "content": system_content}]
        for m in messages:
             if isinstance(m, dict):
                  initial_context.append({"role": m.get("role", "user"), "content": m.get("content", "")})
             else:
                  initial_context.append({"role": m.role, "content": m.content or ""})
        self.context_manager.reset(initial_context)
        
        num_llm_calls_available = self.max_llm_calls
        response = []
        
        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1
            
            # Context Limit Check
            status, effective_messages, meta = self.context_manager.ensure_context_limits()

            newly_pruned_indices = meta.get('newly_pruned_indices', [])
            if newly_pruned_indices:
                print(f"\n\n[Token Limit] Soft Limit exceeded ({meta.get('original_token_count', 'unknown')} > {self.pruning_token_limit}). Pruning {len(newly_pruned_indices)} messages (Turns: {list(sorted(meta.get('excluded_turns', [])))}). New count: {meta.get('token_count')}.")

            is_final_turn = (num_llm_calls_available == 0)

            if status == 'HARD_LIMIT_EXCEEDED' or is_final_turn:
                if status == 'HARD_LIMIT_EXCEEDED':
                    print(f"\n\n[Token Limit] Hard Limit exceeded ({meta.get('token_count', 'unknown')} > {self.max_context_tokens}). Forcing Final Answer.")
                    final_messages = self.context_manager.get_messages()
                else:
                    print(f"\n\n[Max Turns] Final turn reached ({self.max_llm_calls}). Forcing Final Answer.")
                    final_messages = self.context_manager.get_final_turn_messages()

                content, usage = self.call_server(final_messages, max_tries=self.max_retries)
                current_usage = usage if usage else {}
                final_extra = {'turn': self.context_manager.current_turn, 'token_usage': current_usage, 'context_tokens': meta.get('token_count', 0), 'forced_final': 'hard_limit' if status == 'HARD_LIMIT_EXCEEDED' else 'max_turns', 'truncation_notice': (self.context_manager.truncated_msg + '\n' + self.context_manager.final_msg) if status == 'HARD_LIMIT_EXCEEDED' else self.context_manager.final_msg}
                self.context_manager.add_message({"role": "assistant", "content": content.strip(), "extra": final_extra})
                final_msg = Message(role="assistant", content=content.strip(), extra=final_extra)
                response.append(final_msg)
                yield response
                return

            # Call LLM with effective messages
            content, usage = self.call_server(self.context_manager.get_messages(), max_tries=self.max_retries)
            
            if content == "Error: LLM server call failed after all retries.":
                final_msg = Message(role="assistant", content=content, extra={'token_usage': usage if usage else {}})
                response.append(final_msg)
                yield response
                return
            
            current_usage = usage if usage else {}

            # Clean up tool_response leakage
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            
            # Prepare extra metadata (usage + pruning) BEFORE adding to context
            msg_extra = {'turn': self.context_manager.current_turn, 'token_usage': current_usage, 'context_tokens': meta.get('token_count', 0)}
            if newly_pruned_indices:
                msg_extra['newly_pruned_indices'] = list(newly_pruned_indices)

            # Add assistant message to Context Manager WITH extra
            self.context_manager.add_message({
                "role": "assistant",
                "content": content.strip(),
                "extra": msg_extra
            })

            assistant_msg = Message(role="assistant", content=content.strip(), extra=msg_extra)
            response.append(assistant_msg)
            yield response
            
            # Parse tool calls using robust extractor
            tool_calls = robust_extract_tool_calls(content)
            
            if tool_calls:
                # Execute first tool call
                tool_call = tool_calls[0]
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('arguments', {})
                
                # Use the new method to get both string and raw result
                result_str, raw_result = self.custom_call_tool_with_raw(tool_name, tool_args)
                result_wrapped = "<tool_response> Observation:\n\n" + result_str + "\n</tool_response>"
                
                # Prepare extra metadata
                fn_extra = {'turn': self.context_manager.current_turn}
                
                logger.info(f"DEBUG: Tool {tool_name} returned raw_result type: {type(raw_result)}")
                
                # If raw_result is a string (common), try to parse it back to structure
                if isinstance(raw_result, str):
                    try:
                        parsed = json5.loads(raw_result)
                        if isinstance(parsed, (list, dict)):
                            raw_result = parsed
                            logger.info(f"DEBUG: Successfully parsed JSON string for {tool_name}")
                        else:
                            logger.info(f"DEBUG: Parsed result for {tool_name} is not list/dict: {type(parsed)}")
                    except Exception as e:
                        logger.info(f"DEBUG: Failed to parse JSON string for {tool_name}: {e}")
                
                if raw_result is not None and isinstance(raw_result, (list, dict)):
                     fn_extra['raw_tool_response'] = raw_result
                     logger.info(f"DEBUG: Stored raw_tool_response for {tool_name}")
                else:
                     logger.info(f"DEBUG: NOT storing raw_tool_response for {tool_name} (type: {type(raw_result)})")
                
                # Append tool result to Context Manager WITH extra
                self.context_manager.add_message({
                    "role": "user", 
                    "content": result_wrapped,
                    "extra": fn_extra
                })
                
                fn_msg = Message(role=FUNCTION, name=tool_name, content=result_str, extra=fn_extra)
                response.append(fn_msg)
                yield response
                
                self.context_manager.advance_turn()
                continue
            
            # Check for answer tag (termination). Also accept the bare
            # fallback phrase in case the LLM forgot the <answer> tags.
            if _is_termination(content):
                # Valid termination
                self.context_manager.advance_turn()
                break
            
            # No tool and no answer - inject error
            error_msg = ("Error: The response did not contain a valid tool call or an <answer> tag. "
                        "You must either take an action (inside <tool_call></tool_call>) or provide a final answer (inside <answer></answer>).")
            
            self.context_manager.add_message({"role": "user", "content": error_msg})
            
            err_msg = Message(role="user", content=error_msg)
            response.append(err_msg)
            yield response
            self.context_manager.advance_turn()
        
        yield response


class CustomGPT5MultiTurnReactAgent(FnCallAgent):
    """
    Custom GPT-5/O1/O3 Agent using custom call_server() and XML-based tool parsing.
    Uses max_completion_tokens and reasoning_effort for thinking models.
    Removes native OpenAI function calling for fair comparison.
    """
    def __init__(self,
                 llm_config: Dict,
                 system_message: str,
                 function_list: List[BaseTool],
                 verbosity_level: int = 1,
                 max_llm_calls: int = 20,
                 max_retries: int = 5,
                 retry_base_sleep: int = 2,
                 max_context_tokens: int = 110000,
                 pruning_token_limit: int = 100000,
                 prompts_dir: str = None,
                 **kwargs):
        
        # Sanitize system prompt
        sanitized_system_message = system_message
        if "# Tools" in sanitized_system_message:
            sanitized_system_message = sanitized_system_message.split("# Tools")[0].strip()
        
        self.system_message = sanitized_system_message
        self.name = kwargs.get('name')
        self.description = kwargs.get('description')
        
        self.verbosity = verbosity_level
        self.model_name = llm_config.get('model', 'unknown')
        self.model_server = llm_config.get('model_server', None)
        self.api_key = llm_config.get('api_key', "EMPTY")
        self.generate_cfg = llm_config.get('generate_cfg', {})
        self.max_llm_calls = max_llm_calls
        self.max_retries = max_retries
        self.retry_base_sleep = retry_base_sleep
        self.max_context_tokens = max_context_tokens
        self.pruning_token_limit = pruning_token_limit

        self.function_map = {}
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

        self.tool_prompt = generate_tool_prompt_xml(self.function_map)
        self.client = OpenAI(api_key=self.api_key, base_url=self.model_server)

        from agent.context_manager import ContextManager
        self.context_manager = ContextManager(token_limit=self.pruning_token_limit, max_context_tokens=self.max_context_tokens, model_name=self.model_name, prompts_dir=prompts_dir)

    def run(self, messages: List[Union[Dict, Message]], **kwargs) -> Iterator[List[Message]]:
        from qwen_agent.llm.schema import Message
        import copy
        
        messages = copy.deepcopy(messages)
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
        
        for rsp in self._run(messages=new_messages, **kwargs):
            yield rsp

    def call_server(self, msgs: List[Dict], max_tries: int = 10) -> str:
        """Custom OpenAI API call for O1/O3 models with max_completion_tokens."""
        base_sleep_time = self.retry_base_sleep
        
        for attempt in range(max_tries):
            try:

                
                api_kwargs = {
                    "model": self.model_name,
                    "messages": msgs,
                    "max_completion_tokens": self.generate_cfg.get('max_output_tokens', 4096)
                }
                
                # Reasoning effort for O1/O3 models
                if self.generate_cfg.get('reasoning_effort'):
                    api_kwargs["reasoning_effort"] = self.generate_cfg['reasoning_effort']
                
                # Temperature if supported
                if self.generate_cfg.get('temperature') is not None:
                    api_kwargs["temperature"] = self.generate_cfg['temperature']
                
                chat_response = self.client.chat.completions.create(**api_kwargs)
                content = chat_response.choices[0].message.content
                if content and content.strip():
                    # Extract usage
                    usage = getattr(chat_response, 'usage', None)
                    usage_dict = {}
                    if usage:
                        try:
                            usage_dict = usage.model_dump()
                        except (AttributeError, TypeError):
                            try:
                                usage_dict = dict(usage)
                            except (TypeError, AttributeError, ValueError):
                                pass
                        
                        # Ensure standard keys exist
                        usage_dict.setdefault('prompt_tokens', getattr(usage, 'prompt_tokens', 0))
                        usage_dict.setdefault('completion_tokens', getattr(usage, 'completion_tokens', 0))
                        usage_dict.setdefault('total_tokens', getattr(usage, 'total_tokens', 0))

                    return content.strip(), usage_dict
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")
                    
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")
            
            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return "Error: LLM server call failed after all retries.", {}

    def custom_call_tool_with_raw(self, tool_name: str, tool_args: Dict) -> Tuple[str, Any]:
        """Execute tool and return both string output and raw result."""
        if tool_name in self.function_map:
            try:
                raw_result = self.function_map[tool_name].call(tool_args)
                return str(raw_result), raw_result
            except Exception as e:
                err = f"Error executing {tool_name}: {e}"
                return err, None
        else:
            return f"Error: Tool {tool_name} not found", None

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Custom ReAct loop for GPT-5/O1/O3 with XML-based tool parsing."""
        from qwen_agent.llm.schema import FUNCTION, Message
        import copy
        
        # Prepare initial messages for Context Manager with SYSTEM PROMPT
        system_content = self.system_message + generate_tool_prompt_xml(self.function_map)
        
        initial_context = [{"role": "system", "content": system_content}]
        
        for m in messages:
             if isinstance(m, dict):
                  initial_context.append({"role": m.get("role", "user"), "content": m.get("content", "")})
             else:
                  initial_context.append({"role": m.role, "content": m.content or ""})

        self.context_manager.reset(initial_context)
        
        num_llm_calls_available = self.max_llm_calls
        response = []
        
        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1
            
            # Context Limit Check
            status, effective_messages, meta = self.context_manager.ensure_context_limits()

            newly_pruned_indices = meta.get('newly_pruned_indices', [])
            if newly_pruned_indices:
                print(f"\n\n[Token Limit] Soft Limit exceeded ({meta.get('original_token_count', 'unknown')} > {self.pruning_token_limit}). Pruning {len(newly_pruned_indices)} messages (Turns: {list(sorted(meta.get('excluded_turns', [])))}). New count: {meta.get('token_count')}.")

            is_final_turn = (num_llm_calls_available == 0)

            if status == 'HARD_LIMIT_EXCEEDED' or is_final_turn:
                if status == 'HARD_LIMIT_EXCEEDED':
                    print(f"\n\n[Token Limit] Hard Limit exceeded ({meta.get('token_count', 'unknown')} > {self.max_context_tokens}). Forcing Final Answer.")
                    final_messages = self.context_manager.get_messages()
                else:
                    print(f"\n\n[Max Turns] Final turn reached ({self.max_llm_calls}). Forcing Final Answer.")
                    final_messages = self.context_manager.get_final_turn_messages()

                content, usage = self.call_server(final_messages)
                current_usage = usage if usage else {}
                final_extra = {'turn': self.context_manager.current_turn, 'token_usage': current_usage, 'context_tokens': meta.get('token_count', 0), 'forced_final': 'hard_limit' if status == 'HARD_LIMIT_EXCEEDED' else 'max_turns', 'truncation_notice': (self.context_manager.truncated_msg + '\n' + self.context_manager.final_msg) if status == 'HARD_LIMIT_EXCEEDED' else self.context_manager.final_msg}
                self.context_manager.add_message({"role": "assistant", "content": content.strip(), "extra": final_extra})
                final_msg = Message(role="assistant", content=content.strip(), extra=final_extra)
                response.append(final_msg)
                yield response
                return

            # Call LLM with effective messages
            content, usage = self.call_server(self.context_manager.get_messages())
            
            if content == "Error: LLM server call failed after all retries.":
                final_msg = Message(role="assistant", content=content, extra={'token_usage': usage if usage else {}})
                response.append(final_msg)
                yield response
                return
            
            current_usage = usage if usage else {}
            
            # Clean up tool_response leakage
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            # Truncate after first </tool_call> to prevent repeated tool calls
            first_tc_close = content.find('</tool_call>')
            if first_tc_close != -1:
                content = content[:first_tc_close + len('</tool_call>')]

            # Prepare extra metadata (usage + pruning) BEFORE adding to context
            msg_extra = {'turn': self.context_manager.current_turn, 'token_usage': current_usage, 'context_tokens': meta.get('token_count', 0)}
            if newly_pruned_indices:
                msg_extra['newly_pruned_indices'] = list(newly_pruned_indices)

            # Add assistant message to Context Manager WITH extra
            self.context_manager.add_message({
                "role": "assistant",
                "content": content.strip(),
                "extra": msg_extra
            })

            assistant_msg = Message(role="assistant", content=content.strip(), extra=msg_extra)
            response.append(assistant_msg)
            yield response

            # Parse tool calls using robust extractor
            tool_calls = robust_extract_tool_calls(content)
            
            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('arguments', {})
                
                result, raw_result = self.custom_call_tool_with_raw(tool_name, tool_args)
                result_wrapped = "<tool_response> Observation:\n\n" + result + "\n</tool_response>"
                
                # Prepare extra metadata
                fn_extra = {'turn': self.context_manager.current_turn}
                
                if isinstance(raw_result, str):
                    try:
                        parsed = json5.loads(raw_result)
                        if isinstance(parsed, (list, dict)):
                            raw_result = parsed
                    except (ValueError, TypeError):
                        pass
                
                if raw_result is not None and isinstance(raw_result, (list, dict)):
                     fn_extra['raw_tool_response'] = raw_result

                # Append tool result to Context Manager WITH extra
                self.context_manager.add_message({
                    "role": "user", 
                    "content": result_wrapped,
                    "extra": fn_extra
                })

                fn_msg = Message(role=FUNCTION, name=tool_name, content=result, extra=fn_extra)
                response.append(fn_msg)
                yield response
                
                self.context_manager.advance_turn()
                continue
            
            # Check for answer tag (termination). Also accept the bare
            # fallback phrase in case the LLM forgot the <answer> tags.
            if _is_termination(content):
                self.context_manager.advance_turn()
                break
            
            # No tool and no answer - inject error
            error_msg = ("Error: The response did not contain a valid tool call or an <answer> tag. "
                        "You must either take an action (inside <tool_call></tool_call>) or provide a final answer (inside <answer></answer>).")
            
            self.context_manager.add_message({"role": "user", "content": error_msg})
            
            err_msg = Message(role="user", content=error_msg)
            response.append(err_msg)
            yield response
            self.context_manager.advance_turn()
        
        yield response


class CustomGPT5ResponsesReactAgent(FnCallAgent):
    """
    Custom GPT-5/O-series Agent using OpenAI Responses API.
    Uses client.responses.create() with input/instructions format.
    Supports reasoning models with max_output_tokens and reasoning effort.
    """
    def __init__(self,
                 llm_config: Dict,
                 system_message: str,
                 function_list: List[BaseTool],
                 verbosity_level: int = 1,
                 max_llm_calls: int = 20,
                 max_retries: int = 5,
                 retry_base_sleep: int = 2,
                 max_context_tokens: int = 110000,
                 pruning_token_limit: int = 100000,
                 prompts_dir: str = None,
                 **kwargs):
        
        # Sanitize system prompt
        sanitized_system_message = system_message
        if "# Tools" in sanitized_system_message:
            sanitized_system_message = sanitized_system_message.split("# Tools")[0].strip()
        
        self.system_message = sanitized_system_message
        self.name = kwargs.get('name')
        self.description = kwargs.get('description')
        
        self.verbosity = verbosity_level
        self.model_name = llm_config.get('model', 'unknown')
        self.model_server = llm_config.get('model_server', None)
        self.api_key = llm_config.get('api_key', "EMPTY")
        self.generate_cfg = llm_config.get('generate_cfg', {})
        self.max_llm_calls = max_llm_calls
        self.max_retries = max_retries
        self.retry_base_sleep = retry_base_sleep
        self.max_context_tokens = max_context_tokens
        self.pruning_token_limit = pruning_token_limit
        
        self.function_map = {}
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

        self.tool_prompt = generate_tool_prompt_xml(self.function_map)
        self.client = OpenAI(api_key=self.api_key, base_url=self.model_server)

        from agent.context_manager import ContextManager
        self.context_manager = ContextManager(token_limit=self.pruning_token_limit, max_context_tokens=self.max_context_tokens, model_name=self.model_name, prompts_dir=prompts_dir)

    def run(self, messages: List[Union[Dict, Message]], **kwargs) -> Iterator[List[Message]]:
        from qwen_agent.llm.schema import Message
        import copy

        messages = copy.deepcopy(messages)
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)

        for rsp in self._run(messages=new_messages, **kwargs):
            yield rsp

    def call_server(self, input_items: List[Dict], max_tries: int = 10) -> str:
        """Call OpenAI Responses API with input items."""
        base_sleep_time = self.retry_base_sleep
        
        # Build instructions with tool prompt
        instructions = self.system_message + generate_tool_prompt_xml(self.function_map)
        
        for attempt in range(max_tries):
            try:
                api_kwargs = {
                    "model": self.model_name,
                    "input": input_items,
                    "instructions": instructions,
                    "max_output_tokens": self.generate_cfg.get('max_output_tokens', 4096)
                }
                
                # Reasoning effort for O1/O3/GPT-5 models
                if self.generate_cfg.get('reasoning_effort'):
                    api_kwargs["reasoning"] = {"effort": self.generate_cfg['reasoning_effort']}
                
                # Temperature if supported
                if self.generate_cfg.get('temperature') is not None:
                    api_kwargs["temperature"] = self.generate_cfg['temperature']
                
                response = self.client.responses.create(**api_kwargs)
                
                # Extract content from response
                # Responses API returns output as a list of items
                content = ""
                if hasattr(response, 'output_text') and response.output_text:
                    content = response.output_text
                elif hasattr(response, 'output') and response.output:
                    # Parse output items for text content
                    for item in response.output:
                        if hasattr(item, 'type'):
                            if item.type == 'message' and hasattr(item, 'content'):
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content += content_item.text
                            elif item.type == 'text' and hasattr(item, 'text'):
                                content += item.text
                
                if content and content.strip():
                     # Extract usage
                    usage = getattr(response, 'usage', None)
                    usage_dict = {}
                    if usage:
                        try:
                            usage_dict = usage.model_dump()
                        except (AttributeError, TypeError):
                            try:
                                usage_dict = dict(usage)
                            except (TypeError, AttributeError, ValueError):
                                pass
                        
                        # Ensure standard keys exist
                        usage_dict.setdefault('input_tokens', getattr(usage, 'input_tokens', 0))
                        usage_dict.setdefault('output_tokens', getattr(usage, 'output_tokens', 0))
                        usage_dict.setdefault('total_tokens', getattr(usage, 'total_tokens', 0))
                        
                        # Map to standard keys
                        usage_dict['prompt_tokens'] = usage_dict.get('input_tokens', 0)
                        usage_dict['completion_tokens'] = usage_dict.get('output_tokens', 0)
                        usage_dict.setdefault('prompt_tokens', 0)
                        usage_dict.setdefault('completion_tokens', 0)
                        usage_dict.setdefault('total_tokens', 0)

                    return content.strip(), usage_dict
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")
                    
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")
            
            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return "Error: LLM server call failed after all retries.", {}

    def custom_call_tool_with_raw(self, tool_name: str, tool_args: Dict) -> Tuple[str, Any]:
        """Execute tool and return both string output and raw result."""
        if tool_name in self.function_map:
            try:
                raw_result = self.function_map[tool_name].call(tool_args)
                return str(raw_result), raw_result
            except Exception as e:
                err = f"Error executing {tool_name}: {e}"
                return err, None
        else:
            return f"Error: Tool {tool_name} not found", None

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """
        Custom ReAct loop for GPT-5/O-series using Responses API.
        Uses input items format instead of messages array.
        """
        from qwen_agent.llm.schema import FUNCTION, Message
        import copy
        
        # Prepare initial context (standard messages)
        # We store standard messages in ContextManager and convert to input_items for API
        system_content = self.system_message + generate_tool_prompt_xml(self.function_map)
        initial_context = [{"role": "system", "content": system_content}]
        
        for m in messages:
             if isinstance(m, dict):
                  initial_context.append({"role": m.get("role", "user"), "content": m.get("content", "")})
             else:
                  initial_context.append({"role": m.role, "content": m.content or ""})

        self.context_manager.reset(initial_context)
        
        num_llm_calls_available = self.max_llm_calls
        response = []
        
        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1
            
            # Context Limit Check
            status, effective_messages, meta = self.context_manager.ensure_context_limits()

            newly_pruned_indices = meta.get('newly_pruned_indices', [])
            if newly_pruned_indices:
                print(f"\n\n[Token Limit] Soft Limit exceeded ({meta.get('original_token_count', 'unknown')} > {self.pruning_token_limit}). Pruning {len(newly_pruned_indices)} messages (Turns: {list(sorted(meta.get('excluded_turns', [])))}). New count: {meta.get('token_count')}.")

            # Helper to convert messages to input_items
            def to_input_items(msgs):
                items = []
                for m in msgs:
                    role = m.get('role', 'user')
                    content = m.get('content', '')
                    if role == 'system': continue
                    item_role = 'assistant' if role == 'assistant' else 'user'
                    items.append({"type": "message", "role": item_role, "content": content})
                return items

            is_final_turn = (num_llm_calls_available == 0)

            if status == 'HARD_LIMIT_EXCEEDED' or is_final_turn:
                if status == 'HARD_LIMIT_EXCEEDED':
                    print(f"\n\n[Token Limit] Hard Limit exceeded ({meta.get('token_count', 'unknown')} > {self.max_context_tokens}). Forcing Final Answer.")
                    final_messages = self.context_manager.get_messages()
                else:
                    print(f"\n\n[Max Turns] Final turn reached ({self.max_llm_calls}). Forcing Final Answer.")
                    final_messages = self.context_manager.get_final_turn_messages()

                input_items = to_input_items(final_messages)
                content, usage = self.call_server(input_items)
                current_usage = usage if usage else {}
                final_extra = {'turn': self.context_manager.current_turn, 'token_usage': current_usage, 'context_tokens': meta.get('token_count', 0), 'forced_final': 'hard_limit' if status == 'HARD_LIMIT_EXCEEDED' else 'max_turns', 'truncation_notice': (self.context_manager.truncated_msg + '\n' + self.context_manager.final_msg) if status == 'HARD_LIMIT_EXCEEDED' else self.context_manager.final_msg}
                self.context_manager.add_message({"role": "assistant", "content": content.strip(), "extra": final_extra})
                final_msg = Message(role="assistant", content=content.strip(), extra=final_extra)
                response.append(final_msg)
                yield response
                return

            # Normal Execution
            input_items = to_input_items(effective_messages)
            content, usage = self.call_server(input_items)

            if content == "Error: LLM server call failed after all retries.":
                final_msg = Message(role="assistant", content=content, extra={'token_usage': usage if usage else {}})
                response.append(final_msg)
                yield response
                return

            current_usage = usage if usage else {}

            # Clean up tool_response leakage
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            # Truncate after first </tool_call> to prevent repeated tool calls
            first_tc_close = content.find('</tool_call>')
            if first_tc_close != -1:
                content = content[:first_tc_close + len('</tool_call>')]

            # Prepare extra metadata (usage + pruning) BEFORE adding to context
            msg_extra = {'turn': self.context_manager.current_turn, 'token_usage': current_usage, 'context_tokens': meta.get('token_count', 0)}
            if newly_pruned_indices:
                msg_extra['newly_pruned_indices'] = list(newly_pruned_indices)

            # Add assistant message to Context Manager WITH extra
            self.context_manager.add_message({
                "role": "assistant",
                "content": content.strip(),
                "extra": msg_extra
            })

            assistant_msg = Message(role="assistant", content=content.strip(), extra=msg_extra)
            response.append(assistant_msg)
            yield response

            # Parse tool calls using robust extractor
            tool_calls = robust_extract_tool_calls(content)

            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('arguments', {})

                result, raw_result = self.custom_call_tool_with_raw(tool_name, tool_args)
                result_wrapped = "<tool_response> Observation:\n\n" + result + "\n</tool_response>"

                # Prepare extra metadata
                fn_extra = {'turn': self.context_manager.current_turn}

                if isinstance(raw_result, str):
                    try:
                        parsed = json5.loads(raw_result)
                        if isinstance(parsed, (list, dict)):
                            raw_result = parsed
                    except (ValueError, TypeError):
                        pass

                if raw_result is not None and isinstance(raw_result, (list, dict)):
                     fn_extra['raw_tool_response'] = raw_result

                # Append tool result to Context Manager WITH extra
                self.context_manager.add_message({
                    "role": "user",
                    "content": result_wrapped,
                    "extra": fn_extra
                })

                fn_msg = Message(role=FUNCTION, name=tool_name, content=result, extra=fn_extra)
                response.append(fn_msg)
                yield response

                self.context_manager.advance_turn()
                continue

            # Check for answer tag (termination). Also accept the bare
            # fallback phrase in case the LLM forgot the <answer> tags.
            if _is_termination(content):
                self.context_manager.advance_turn()
                break

            # No tool and no answer - inject error
            error_msg = ("Error: The response did not contain a valid tool call or an <answer> tag. "
                        "You must either take an action (inside <tool_call></tool_call>) or provide a final answer (inside <answer></answer>).")

            self.context_manager.add_message({"role": "user", "content": error_msg})

            err_msg = Message(role="user", content=error_msg)
            response.append(err_msg)
            yield response
            self.context_manager.advance_turn()

        yield response
