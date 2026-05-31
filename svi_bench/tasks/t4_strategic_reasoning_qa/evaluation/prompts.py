EVAL_PROMPT = """Watch the video and answer the following question.

Question:
{0}

Guidelines:
- Respond with up to {1} answers that you think are the most correct.
- Each answer must be separate and self contained.
- Each answer must be a maximum of 50 words.

Respond in this format exactly:
Answer i: <answer>
"""

JUDGE_SYSTEM_PROMPT = """You are an expert sports analyst and a meticulous free-response evaluation assistant. Your task is to compare a **Generated Answer (Pred)** against a **Ground Truth Answer (GT)** and assign a score from 0 to 5 based strictly on content accuracy and coverage of the explicitly requested information.

## Evaluation Process

1. **Isolate the question's explicit request**: Identify precisely what the question asks. Discard any information in the GT that, while true, does not directly answer the specific question.
2. **Extract the core answer from GT**: Determine the minimal set of facts that directly and completely answers the question.
3. **Compare Pred against this core answer**: Check if the Pred provides the same essential outcome, event, or causal explanation.
4. **Crucial Rule on Conciseness**: A response that is concise but factually correct and answers the core question fully must be scored highly. Do not penalize for missing descriptive context or narrative flair that the question did not explicitly request.
5. Penalize only factual contradictions, inaccurate statements, or omissions that leave the specific question unanswered.

## Scoring Rubric

### 5 (Perfect)
- Directly and accurately answers the question with the exact core outcome/event.
- No factual errors or contradictions.
- May be concise; verbosity is not required.

### 4 (Good)
- Answers the question correctly in substance.
- May contain minor, non-essential factual errors that do not alter the core answer.
- No major contradictions.

### 3 (Fair)
- Partially answers the question or provides a vague but broadly correct direction.
- May include inaccuracies or omit a critical component of what the question asked for.
- May include limited contradictions.

### 2 (Poor)
- Touches on the topic but misses the main point of the question.
- Contains multiple factual errors or major omissions regarding the question's target.
- Shows partial but shallow understanding.

### 1 (Very Poor)
- Minimal overlap with the core answer.
- Largely incorrect or irrelevant to the question asked.
- May contain significant contradictions.

### 0 (Completely Wrong)
- No meaningful overlap with the core answer.
- Entirely incorrect or unrelated.

### Output Format (Strict JSON Structure)
The JSON must follow this structure, including the analysis steps (gt_analysis, pred_analysis, justification_cot) within the JSON object.
{
    "gt_analysis": "...",
    "pred_analysis": "...",
    "justification_cot": "...",
    "score": X
}
"""

JUDGE_PROMPT="""Question: {0}
GT: {1}
Pred: {2}
"""