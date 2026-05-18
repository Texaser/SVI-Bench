from gpt import *
import json

SYSTEM_PROMPT = """You are a sports video understanding expert. Given a question and answer, your task is to revise the question such that it does not give away the answer, while retaining certain context that makes it answerable.

### Examples

**Example 1**
Question: In the opening stretch, what repeated offensive priority does Boston use to bend Cleveland\u2019s interior defense before generating higher-value shots, and what kind of secondary shot does this reliably create once the defense collapses?
Answer: Boston repeatedly drives into the paint to force help, then swings the ball quickly side-to-side. Once the defense collapses, the follow-up is open perimeter shooting\u2014especially spot-up threes created by extra passes out of penetration.
Revised: In the opening stretch, what repeated offensive priority does Boston use, what effect does it have on Cleveland's defense, and what kind of shot does Boston reliably create from it?

**Example 2**
Question: Assess Elijah Pepper\u2019s offensive versatility in the first half: would you grade it good, fair, or poor, and which different scoring methods shown on screen most support your grade?"
Answer: Good. He scores as a spot-up three-point shooter, creates separation for midrange, and finishes at the rim off the dribble. The variety forces different defensive answers, and he sustains production rather than relying on only one shot type.
Revised: Assess Elijah Pepper\u2019s offensive versatility in the first half: would you grade it good, fair, or poor, and why?

**Example 3**
Question: On the first-quarter transition where Panathinaikos committed a hard shooting foul to stop a breakaway, what earlier decision and defensive action most directly caused that foul situation to exist in the first place?
Answer: A Panathinaikos perimeter attack was disrupted by a passing-lane read that created a Monaco breakaway. The defender then fouled in the act of shooting to prevent an uncontested finish. The foul was a consequence of the live-ball steal and immediate one-on-one transition.
Revised: On the first-quarter play where Panathinaikos committed a hard shooting foul, why did they do it and what was the context that led up to it?

**Example 4**
Question: What was unusual about the way Valparaiso\u2019s Quentin Green accumulated three fouls in a very short span, and what immediate impact did it have on his availability in the first half?
Answer: He picked up his second personal foul and then immediately got a technical for his reaction with the ball, giving him two fouls at once. He went to the bench and did not return for the remainder of the first half.
Revised: What was unusual about the way Valparaiso\u2019s Quentin Green\u2019's second personal foul? Was there an impact on his availability in the first half?

**Example 5**
Question: On Orlando\u2019s successful backdoor scores in the first half, what off-ball defensive lapse created the lane, and which area of the floor did the cutter use to get behind the defense?
Answer: Denver\u2019s weak-side defenders ball-watched and didn\u2019t track cutters behind them. The cutter repeatedly slipped along the baseline and behind the help defense, turning what looked like contained perimeter action into uncontested finishes at the rim.
Revised: On similar Orlando finishes at the rim in the first half, what was the cause and what area of the floor did Orlando\u2019s finishers use on their way to the basket?

You must respond in this form:
Revised question: ...
"""

PROMPT = """Question: {0}
Answer: {1}
"""

def parse_response(response):
    revised = re.findall(r"Revised question:[\s]*(.+)", response)
    if len(revised) > 0:
        return revised[0].strip()
    else:
        return None

def revise(qa_file):
    with open(qa_file, 'r') as f:
        qa = json.load(f)

    i = 0
    messages = []

    for item in qa:
        question = item["question"]
        answer = item["answer"]

        formatted_prompt = PROMPT.format(question, answer)

        messages.append(batch_object(formatted_prompt, "gpt-5.2", id=i, system_prompt=SYSTEM_PROMPT))

        i += 1

    return messages

def parse_responses(qa_file, batch_files):
    with open(qa_file, 'r') as f:
        qa = json.load(f)

    results = []
    for batch_file in batch_files:
        results += parse_batch(batch_file, cost=True)
    results.sort(key= lambda x: int(x[0].split("task-")[-1])) # will be ordered by custom id

    total_cost = 0
    out = []

    for result in results:
        id, response, cost = result[0], result[1], result[2]
        i = int(id.split("task-")[-1])
        total_cost += cost

        revised = parse_response(response)
        qa[i]["question"] = revised
        out.append(qa[i])

    return out, total_cost