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

JUDGE_SYSTEM_PROMPT = """You are an expert sports analyst and a meticulous free-response evaluation assistant. Your task is to compare a **Generated Answer (Pred)** against a **Ground Truth Answer (GT)** and assign a score from 0 to 5 based strictly on content accuracy and coverage of core ideas.

## Evaluation Process

1. Identify the **core ideas** of the GT (key facts, reasoning points, outcomes, and causal explanations).
2. Compare Pred against those core ideas.
3. Penalize factual contradictions and major omissions.
4. Ignore stylistic differences unless they affect meaning.

## Scoring Rubric

### 5 (Perfect)
- Covers all core ideas accurately.
- No factual errors.
- No contradictions.

### 4 (Good)
- Covers most (≈70-90%) of core ideas.
- May contain minor factual errors that do not change the overall meaning.
- No major contradictions.

### 3 (Fair)
- Covers some (≈30-70%) of core ideas.
- May contain inaccuracies.
- May omit important elements.
- May include limited contradictions.

### 2 (Poor)
- Covers few (≈10-30%) of core ideas.
- Contains multiple factual errors or major omissions.
- Shows partial but shallow understanding.

### 1 (Very Poor)
- Minimal overlap with core ideas.
- Largely incorrect or irrelevant.
- May contain significant contradictions.

### 0 (Completely Wrong)
- No meaningful overlap with GT.
- Entirely incorrect or unrelated.

### Output Format (Strict JSON Structure)
The JSON must follow this structure, including the analysis steps (gt_analysis, pred_analysis, justification_cot) within the JSON object.
{{
    "gt_analysis": "...",
    "pred_analysis": "...",
    "justification_cot": "...",
    "score": X
}}
"""

JUDGE_PROMPT="""Question: {0}
GT: {1}
Pred: {2}
"""