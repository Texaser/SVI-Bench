from openai import OpenAI
import json
import re

API_KEY = None
DEEPSEEK_API_KEY = None

COSTS = {
    'gpt-5': {
        'input': 1.25e-6,
        'output': 1e-5
    },
    'gpt-5.1': {
        'input': 1.25e-6,
        'output': 1e-5
    },
    'gpt-5.2': {
        'input': 1.75e-6,
        'output': 1.4e-5
    },
    'gpt-5-mini': {
        'input': 2.5e-7,
        'output': 2e-6
    },
    'deepseek-chat': {
        'input': 2.8e-7,
        'output': 4.2e-7
    },
    'deepseek-reasoner': {
        'input': 2.8e-7,
        'output': 4.2e-7
    }
}

def gpt_structured(prompt, model, format, reasoning=None, system_prompt="You are a helpful assistant."):
    client = OpenAI(api_key=API_KEY)

    response = client.responses.parse(
        model=model,
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        text_format=format
    ) if reasoning == None else client.responses.parse(
        model=model,
        input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        reasoning={"effort": reasoning},
        text_format=format
    )

    response = response.output_parsed

    return response

def gpt_response(prompt, model):
    client = OpenAI(api_key=API_KEY)
    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text

def deepseek(prompt, model, system_prompt="You are a helpful assistant.", json=False):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={
            'type': 'json_object'
        },
        stream=False
    ) if json else client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )

    return response.choices[0].message.content


def gpt(prompt, model, reasoning=None, cost=False, system_prompt="You are a helpful assistant."):
    client = OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=False
    ) if reasoning == None else client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        reasoning_effort=reasoning,
        stream=False
    )

    if cost:
        input_cost = response.usage.prompt_tokens * COSTS[model]['input']
        output_cost = response.usage.completion_tokens * COSTS[model]['output']
        return (response.choices[0].message.content, input_cost+output_cost)

    return response.choices[0].message.content

def video_gpt(prompt, frames, model):
    client = OpenAI(api_key=API_KEY)
    response = client.responses.create(
        model=model,
        input = [
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": (prompt) },
                    *[
                        { "type": "input_image", "image_url": f"data:image/jpeg;base64,{frame}" }
                        for frame in frames
                    ]
                ]
            }
        ]
    )

    usage = response.usage
    cost = COSTS[model]['input']*usage.input_tokens + COSTS[model]['output']*usage.output_tokens

    return response.output_text, cost

def list_batches():
    client = OpenAI(api_key=API_KEY)
    list_of_batches = client.batches.list()
    for batch in list_of_batches:
        print(f"Batch ID: {batch.id}, Status: {batch.status}")

def batch_gpt(file, endpoint):
    client = OpenAI(api_key=API_KEY)

    batch_input_file = client.files.create(
        file=open(file, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    # print(f"Uploaded file ID: {batch_input_file_id}")

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint=endpoint,
        completion_window="24h"
    )

    batch_job_id = batch_job.id
    print(f'"{batch_job_id}",')

def get_batch(id, file="batch.jsonl"):
    client = OpenAI(api_key=API_KEY)
    batch_job = client.batches.retrieve(id)

    if batch_job.status == "completed":
        result_file_id = batch_job.output_file_id
        result_content = client.files.content(result_file_id).content
        with open(file, 'wb') as f:
            f.write(result_content)
    
def parse_batch(file, cost=False):
    with open(file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
            
    responses = []

    for result in results:
        #response = result['response']['body']['choices'][0]['message']['content']

        try:
            response = result['response']['body']['output'][-1]['content'][0]['text']
        except:
            continue
        custom_id = result['custom_id']

        if cost:
            model = 'gpt-5-mini' if 'mini' in result['response']['body']['model'] else 'gpt-5'
            input_cost = result['response']['body']['usage']['input_tokens'] * COSTS[model]['input']
            output_cost = result['response']['body']['usage']['output_tokens'] * COSTS[model]['output']
            responses.append((custom_id, response, (input_cost+output_cost)/2))
        else:
            responses.append((custom_id, response))

    return responses

def to_schema(format):

    schema = format.model_json_schema()

    def helper(schema):
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                schema["additionalProperties"] = False
            for key in schema:
                helper(schema[key])
        if isinstance(schema, list):
            for item in schema:
                helper(item)

    helper(schema)
    return schema
        
def batch_object(prompt, model, id, system_prompt=None, reasoning=None, format=None):
    body = {
        "model": model,
        "input": prompt
    } if system_prompt == None else {
        "model": model,
        "instructions": system_prompt,
        "input": prompt
    }

    if reasoning != None:
        body["reasoning"] = {"effort": reasoning}
    if format != None:
        schema = to_schema(format)
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "response",
                "strict": True,
                "schema": schema
            }
        }

    return {
        "custom_id": f"task-{id}", 
        "method": "POST", 
        "url": "/v1/responses", 
        "body": body
    }

def check_batch(id):
    client = OpenAI(api_key=API_KEY)
    batch_job = client.batches.retrieve(id)
    # if batch_job.error_file_id:
    #     print(client.files.content(batch_job.error_file_id).read().decode("utf-8"))
        
    print(batch_job)

def cancel_batch(id):
    client = OpenAI(api_key=API_KEY)
    client.batches.cancel(id)
