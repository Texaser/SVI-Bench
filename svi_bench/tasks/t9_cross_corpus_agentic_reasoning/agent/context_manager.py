import os
import sys
import copy
import json
from typing import List, Dict, Tuple, Set, Union, Any

# Try imports for token counting
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# Per-message overhead for chat formatting (e.g. ChatML: <|im_start|>role\n...<|im_end|>\n)
PER_MESSAGE_OVERHEAD_TOKENS = 4

# Cache HF tokenizers so we don't re-instantiate on every count.
_HF_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_hf_tokenizer(name_or_path: str):
    """Return a cached HF AutoTokenizer for the given model id / path, or
    None if transformers isn't available or the load fails."""
    if name_or_path in _HF_TOKENIZER_CACHE:
        return _HF_TOKENIZER_CACHE[name_or_path]
    if not _TRANSFORMERS_AVAILABLE:
        _HF_TOKENIZER_CACHE[name_or_path] = None
        return None
    try:
        tok = AutoTokenizer.from_pretrained(name_or_path)
    except Exception:
        tok = None
    _HF_TOKENIZER_CACHE[name_or_path] = tok
    return tok


def _count_text_tokens(text: str, model_name: str = "gpt-4o", tokenizer_path: str = None) -> int:
    """Count tokens for a raw text string.

    Resolution order:
      1. Explicit ``tokenizer_path`` via HF AutoTokenizer.
      2. ``model_name`` looking HF-shaped (contains ``/``) via AutoTokenizer.
      3. ``model_name`` via tiktoken's registry (GPT family).
      4. ``cl100k_base`` tiktoken fallback.
      5. ``len(text) // 4`` heuristic (no tokenizer available).
    Tokenizers are cached so repeated calls are cheap.
    """
    if tokenizer_path:
        tok = _get_hf_tokenizer(tokenizer_path)
        if tok is not None:
            return len(tok.encode(text))

    # HF-shaped model_name (e.g. "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8")
    if model_name and '/' in model_name:
        tok = _get_hf_tokenizer(model_name)
        if tok is not None:
            return len(tok.encode(text))

    if _TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    return len(text) // 4

def count_tokens(messages: List[Dict], model_name: str = "gpt-4o", tokenizer_path: str = None) -> int:
    """Count tokens in a list of messages, including per-message formatting overhead."""
    full_text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content:
            full_text += f"{role}: {content}\n"

    text_tokens = _count_text_tokens(full_text, model_name, tokenizer_path)
    overhead = len(messages) * PER_MESSAGE_OVERHEAD_TOKENS
    return text_tokens + overhead


def load_token_limit_prompts(prompts_dir: str) -> Tuple[str, str]:
    """Load TRUNCATED_MESSAGE and FINAL_MESSAGE from prompts file."""
    prompts_file = os.path.join(prompts_dir, "token_limit_prompts.txt")
    truncated_msg = ""
    final_msg = ""

    if os.path.exists(prompts_file):
        with open(prompts_file, 'r') as f:
            content = f.read()

        # Parse sections
        if "[TRUNCATED_MESSAGE]" in content:
            parts = content.split("[TRUNCATED_MESSAGE]")
            if len(parts) > 1:
                section = parts[1]
                if "[FINAL_MESSAGE]" in section:
                    truncated_msg = section.split("[FINAL_MESSAGE]")[0].strip()
                else:
                    truncated_msg = section.strip()

        if "[FINAL_MESSAGE]" in content:
            parts = content.split("[FINAL_MESSAGE]")
            if len(parts) > 1:
                final_msg = parts[1].strip()

    if not truncated_msg:
        truncated_msg = "--- Maximum Length Limit Reached ---\nThe response is truncated."
    if not final_msg:
        final_msg = "--- Final Step Reached ---\nYou must offer your final answer now."

    return truncated_msg, final_msg



class ContextManager:
    """Manages the context window by truncating (hard limit) when token count exceeds the threshold.
    Preserves the system prompt and initial user query.
    Supports final-turn detection to force a concluding answer."""

    def __init__(self, token_limit: int = 120000, max_context_tokens: int = 128000, model_name: str = 'gpt-4o', prompts_dir: str = None, verbose: bool = False, function_tokens: int = 0):
        self.token_limit = token_limit
        self.max_context_tokens = max_context_tokens
        self.model_name = model_name
        self.verbose = verbose
        self.function_tokens = function_tokens  # Token overhead from function/tool definitions passed separately to API

        if prompts_dir:
            self.truncated_msg, self.final_msg = load_token_limit_prompts(prompts_dir)
        else:
            self.truncated_msg = "--- Maximum Length Limit Reached ---\nThe response is truncated."
            self.final_msg = "--- Final Step Reached ---\nYou must offer your final answer now."

        self.messages: List[Union[Dict, Any]] = []
        self.current_turn: int = 1
        self._cached_effective: List[Union[Dict, Any]] = None
        self._cached_status: str = None
        self._cached_meta: Dict = None

    def reset(self, initial_messages: List[Union[Dict, Any]]) -> None:
        """Resets the context with initial messages."""
        self.messages = []
        self.current_turn = 1
        self._invalidate_cache()
        for m in initial_messages:
            self.add_message(m, increment_turn=False)

    def add_message(self, message: Union[Dict, Any], increment_turn: bool = False) -> None:
        """Adds a message to the full history with turn tracking.

        For dict messages: makes a shallow copy of the incoming dict (and its
        ``extra`` sub-dict, if present) so we never mutate the caller's
        object. Callers can re-use / mutate the input dict freely after this
        returns without corrupting the manager's history.
        """
        if isinstance(message, dict):
            incoming_role = message.get('role')
            has_assistant = any(m.get('role') == 'assistant' for m in self.messages)
            is_initial_setup = (self.current_turn == 1 and not increment_turn and not has_assistant and incoming_role != 'assistant')
            target_turn = 0 if is_initial_setup else self.current_turn

            # Shallow-copy the dict (and its 'extra' sub-dict) before mutating.
            message = dict(message)
            existing_extra = message.get('extra') or {}
            message['extra'] = dict(existing_extra)
            if 'turn' not in message['extra']:
                message['extra']['turn'] = target_turn

        self.messages.append(message)
        self._invalidate_cache()

        if increment_turn:
            self.advance_turn()

    def advance_turn(self) -> None:
        """Increments the turn counter."""
        self.current_turn += 1
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidates cached effective messages so they are recomputed on next access."""
        self._cached_effective = None
        self._cached_status = None
        self._cached_meta = None

    def get_messages(self) -> List[Union[Dict, Any]]:
        """Returns the EFFECTIVE (truncated if needed) list of messages for the API.
        Uses cached result from the most recent ensure_context_limits() call if available.

        Returns a shallow copy of the cached list — callers can append /
        reorder without corrupting the cache.
        """
        if self._cached_effective is not None:
            return list(self._cached_effective)
        _, effective_messages, _ = self.ensure_context_limits()
        return list(effective_messages)

    def get_final_turn_messages(self) -> List[Union[Dict, Any]]:
        """Returns effective messages with the final-turn instruction appended as a user message.
        Used when the agent is on its last available turn and must produce a final answer."""
        effective = list(self.get_messages())
        effective.append({"role": "user", "content": self.final_msg})
        return effective

    def get_full_history(self) -> List[Union[Dict, Any]]:
        """Returns the complete message history (shallow copy, non-destructive)."""
        return list(self.messages)

    def ensure_context_limits(self) -> Tuple[str, List[Union[Dict, Any]], Dict]:
        """Ensures context fits within hard limit. Returns (status, effective_messages, metadata).

        - Soft limit (token_limit): if enabled (!= -1) and exceeded, prunes oldest turns.
        - Hard limit (max_context_tokens): if enabled (!= -1) and exceeded after pruning,
          truncates the last message and returns HARD_LIMIT_EXCEEDED status.
        """
        current_tokens = self._robust_count_tokens(self.messages)

        pruned_indices = set()
        excluded_turns = set()
        effective_messages = self.messages
        new_tokens = current_tokens
        limit_status = 'OK'

        # 1. Soft Limit: prune oldest turns
        if self.token_limit != -1 and current_tokens > self.token_limit:
            effective_messages, pruned_indices, excluded_turns = self.prune_context()
            new_tokens = self._robust_count_tokens(effective_messages)
            limit_status = 'PRUNED'

        # 2. Hard Limit: truncate last message
        if self.max_context_tokens != -1 and new_tokens > self.max_context_tokens:
             if self.verbose:
                print(f"[ContextManager] Hard Limit Exceeded: {new_tokens} > {self.max_context_tokens}")

             effective_messages = self.truncate_to_fit(effective_messages)
             new_tokens = self._robust_count_tokens(effective_messages)

             self._cached_effective = effective_messages
             self._cached_status = 'HARD_LIMIT_EXCEEDED'
             self._cached_meta = {
                 'token_count': new_tokens,
                 'original_token_count': current_tokens,
                 'newly_pruned_indices': list(pruned_indices),
                 'excluded_turns': list(excluded_turns)
             }
             return self._cached_status, self._cached_effective, self._cached_meta

        self._cached_effective = effective_messages
        self._cached_status = limit_status
        self._cached_meta = {
            'newly_pruned_indices': list(pruned_indices),
            'excluded_turns': list(excluded_turns),
            'token_count': new_tokens,
            'original_token_count': current_tokens
        }
        return self._cached_status, self._cached_effective, self._cached_meta

    def prune_context(self) -> Tuple[List[Union[Dict, Any]], Set[int], Set[int]]:
        """Returns the effective subset of messages that fits within the soft token limit."""
        if not self.messages:
            return [], set(), set()

        msgs = self.messages
        current_tokens = self._robust_count_tokens(msgs)

        if current_tokens <= self.token_limit:
            return msgs, set(), set()

        if self.verbose:
            print(f"[ContextManager] Token count {current_tokens} exceeds limit {self.token_limit}. Pruning...")

        indices_to_remove = set()
        excluded_turns = set()

        kept_indices = list(range(len(msgs)))

        while current_tokens > self.token_limit:
            start_index = -1
            candidate_start_idx = -1

            for i in range(len(kept_indices)):
                real_idx = kept_indices[i]
                if real_idx < 2:
                    continue
                if self._get_role(msgs[real_idx]) == 'assistant':
                    candidate_start_idx = i
                    start_index = real_idx
                    break

            if start_index == -1:
                if self.verbose:
                    print("[ContextManager] No more removable turns found.")
                break

            candidate_end_idx = len(kept_indices)
            for i in range(candidate_start_idx + 1, len(kept_indices)):
                if self._get_role(msgs[kept_indices[i]]) == 'assistant':
                    candidate_end_idx = i
                    break

            turn_num = self._get_turn(msgs[start_index])
            if turn_num is not None:
                excluded_turns.add(turn_num)

            for idx in kept_indices[candidate_start_idx:candidate_end_idx]:
                indices_to_remove.add(idx)

            kept_indices = kept_indices[:candidate_start_idx] + kept_indices[candidate_end_idx:]

            effective_msgs = [msgs[i] for i in kept_indices]
            current_tokens = self._robust_count_tokens(effective_msgs)

            if self.verbose:
                print(f"[ContextManager] New token count: {current_tokens}")

        return [msgs[i] for i in kept_indices], indices_to_remove, excluded_turns

    def truncate_to_fit(self, messages: List[Union[Dict, Any]]) -> List[Union[Dict, Any]]:
        """Truncate the last message to fit within max_context_tokens. Returns a new list."""
        if not messages:
            return messages

        effective_msgs = list(messages)

        history_tokens = self._robust_count_tokens(effective_msgs[:-1])
        notice_text = "\n\n" + self.truncated_msg + "\n\n" + self.final_msg
        notice_tokens = self._robust_count_tokens([{'role': 'user', 'content': notice_text}])

        remaining_budget = self.max_context_tokens - history_tokens - notice_tokens - 50

        last_msg = copy.deepcopy(effective_msgs[-1])
        content = self._get_content(last_msg)

        if remaining_budget <= 0:
            last_content = notice_text
        else:
            full_last_content_tokens = self._robust_count_tokens([last_msg])
            if history_tokens + full_last_content_tokens + notice_tokens <= self.max_context_tokens:
                last_content = content + notice_text
            else:
                safe_chars = int(remaining_budget * 3.5)
                last_content = content[:safe_chars] + notice_text

        self._set_content(last_msg, last_content)
        effective_msgs[-1] = last_msg

        return effective_msgs

    def _robust_count_tokens(self, messages: List[Union[Dict, Any]]) -> int:
        """Count tokens including all message fields (content, name, function_call) + function definition overhead."""
        temp_msgs = []
        for m in messages:
            role = self._get_role(m)
            parts = []

            content = self._get_content(m)
            if content:
                parts.append(content)

            name = self._get_name(m)
            if name:
                parts.append(f"name: {name}")

            function_call = self._get_function_call(m)
            if function_call:
                parts.append(f"function_call: {function_call}")

            temp_msgs.append({"role": role, "content": "\n".join(parts)})

        return count_tokens(temp_msgs, self.model_name) + self.function_tokens

    @staticmethod
    def compute_function_tokens(function_defs: list, model_name: str = "gpt-4o") -> int:
        """Compute token overhead for function/tool definitions passed separately to the API."""
        if not function_defs:
            return 0
        text = json.dumps(function_defs)
        return _count_text_tokens(text, model_name)

    def _get_role(self, msg: Union[Dict, Any]) -> str:
        if isinstance(msg, dict):
            return msg.get('role', '')
        return getattr(msg, 'role', '')

    def _get_content(self, msg: Union[Dict, Any]) -> str:
        if isinstance(msg, dict):
            return msg.get('content', '')
        return getattr(msg, 'content', '') or ""

    def _get_name(self, msg: Union[Dict, Any]) -> str:
        if isinstance(msg, dict):
            return msg.get('name', '')
        return getattr(msg, 'name', '') or ''

    def _get_function_call(self, msg: Union[Dict, Any]) -> str:
        if isinstance(msg, dict):
            fc = msg.get('function_call')
        else:
            fc = getattr(msg, 'function_call', None)
        if not fc:
            return ''
        if isinstance(fc, dict):
            return json.dumps(fc)
        try:
            return f"{fc.name}({fc.arguments})"
        except Exception:
            return str(fc)

    def _set_content(self, msg: Union[Dict, Any], content: str) -> None:
        if isinstance(msg, dict):
            msg['content'] = content
        else:
            setattr(msg, 'content', content)

    def _get_turn(self, msg: Union[Dict, Any]) -> Any:
        if isinstance(msg, dict):
            return msg.get('extra', {}).get('turn')
        extra = getattr(msg, 'extra', {})
        if isinstance(extra, dict):
            return extra.get('turn')
        return None
