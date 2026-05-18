import json
from gpt import *
import re

SYSTEM_PROMPT = """You are a sports video understanding expert. Given a question, answer, and evidence determine if all of the following criteria are met.

**Criteria**
- The question and answer only involve a single game, not multiple.
- The question and answer only reference in game information, not interviews, game reports, etc.
- The question and answer do not reference any commentary of the game.
- The answer is COMPLETELY supported by the evidence.

You must respond in this form:

Decision: [Yes/No]
Explanation: [Reasoning]
"""

PROMPT = """Question: {0}

Answer: {1}

Subtitle evidence:
{2}

Game report evidence:
{3}
"""

def parse_response(response):
    decision = re.search(r"Decision:(.*)", response)

    if decision:
        decision = decision.group(1).strip()
        explanation = response.split("Explanation:")[-1].strip()

        return decision.lower() == "yes", explanation
    else:
        print(response)
        return False, ""

def supported(path):
    with open(path, 'r') as f:
        qas = json.load(f)

    id = 0
    messages = []
    supported = []

    for qa in qas:
        question = qa["question"]
        answer = qa["answer"]

        subtitle_evidence = ""
        previous = 0
        for item in qa["subtitle_evidence"]:

            period = item["period"]
            if period != previous:
                subtitle_evidence += "Period {period}:\n"
                previous = period
            
            subtitle_evidence += item.get("subtitles", "")
        
        report_evidence = ""
        for quote in qa["report_evidence"]:
            report_evidence += f'"{quote}"\n'

        formatted = PROMPT.format(question, answer, subtitle_evidence, report_evidence)

        messages.append(batch_object(formatted, "gpt-5-mini", id, system_prompt=SYSTEM_PROMPT))

        id += 1
    
    return messages

def parse_responses(qa_file, batch_file):
    with open(qa_file, 'r') as f:
        qa = json.load(f)

    results = parse_batch(batch_file, cost=True)

    results.sort(key= lambda x: int(x[0].split("task-")[-1])) # will be ordered by custom id

    supported = []

    total_cost = 0

    for result in results:
        custom_id, response, cost = result[0], result[1], result[2]
        i = int(custom_id.split("-")[-1])

        total_cost += cost
        decision, explanation = parse_response(response)

        if decision == True:
            supported.append(qa[i])
    
    return supported, total_cost