import json
import re
import gpt
import gemini
from concurrent.futures import ThreadPoolExecutor

free_response = """Answer the following question.

Question:
{0}

Guidelines:
- Respond with up to 5 answers that you think are the most correct.
- Each answer must be separate and self contained.
- Each answer must be a maximum of 50 words.

Respond using this format:
Answer i: <answer>"""

def parse_answers(text):
    answers = [a.strip() for a in re.findall(r"Answer [1-5]:\s*(.+?)(?=Answer [1-5]:|$)", text, re.DOTALL)]
    return answers


def gpt_blind_filter(model, qa_file):
    with open(qa_file, 'r') as f:
        qa = json.load(f)

    messages = []
    id = 0

    for item in qa:
        question = item['question']

        message = gpt.batch_object(free_response.format(question), model, id=id)
        messages.append(message)
        id += 1

    return messages

def process_item(item):

    if len(item.get('responses', dict()).get('gemini-3-flash-blind', [])) > 0:
        print("skipped")
        return item['responses']['gemini-3-flash-blind']

    question = item['question']

    response = gemini.gemini("gemini-3-flash-preview", free_response.format(question))
    answers = parse_answers(response)

    return answers

def gemini_blind_filter(qa_file, out_file):
    with open(qa_file, 'r') as f:
        qa = json.load(f)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_item, item) for item in qa]

        for i in range(len(futures)):
            if 'responses' not in qa[i]:
                qa[i]['responses'] = dict()
            try:
                qa[i]['responses']['gemini-3-flash-blind'] = futures[i].result()
            except Exception as e:
                print(e)

    with open(out_file, 'w') as f:
        json.dump(qa, f, indent=4)


def parse_responses(qa_file, batch_files):
    with open(qa_file, 'r') as f:
        qa = json.load(f)

    results = []
    for batch_file in batch_files:
        results += gpt.parse_batch(batch_file, cost=True)
    results.sort(key= lambda x: int(x[0].split("task-")[-1])) # will be ordered by custom id

    total_cost = 0
    for result in results:
        id, response, cost = result[0], result[1], result[2]
        i = int(id.split("task-")[-1])
        total_cost += cost

        answers = parse_answers(response)
        
        if 'responses' not in qa[i]:
            qa[i]['responses'] = dict()
        qa[i]['responses']['gpt-5.2-blind'] = answers

    print(total_cost)

    with open('gpt_blind.json', 'w') as f:
        json.dump(qa, f, indent=4)
