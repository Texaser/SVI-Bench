prompt_tactical_strategic_analysis='''I want you to test students' ability to perform tactical and strategic analysis. Some example questions include:
- During [Timeframe], what is [Entity]’s most frequently used [Strategic Metadata] (e.g., offensive play type, defensive scheme) and what was the effectiveness?
- Analyze [Entity]’s performance when facing [Opponent Strategic Metadata] during [Timeframe]. What specific breakdown in [Skill_Set] execution, particularly in [Spatial Information] (e.g., zone, court area), causes the most significant decline in their [Stat_Category]?
- What tactical adjustment did [Entity] make during [In-Game Period] (e.g., halftime, timeout)? Analyze the result.
- Describe the principles of the [Strategic Metadata] (e.g., 2-3 Zone, 4-4-2, Neutral Zone Trap) employed by [Entity] during [Timeframe].
- Compare [Entity]’s performance when [Entity] was on the field versus off the field during [Timeframe].'''

prompt_role_skill_assessment='''I want you to test students' ability to perform player role and skill assessment. Some example questions include:
- What performance grade (good, fair, poor) best matches [Entity]'s execution of [Skill_Set] during [In-Game Period] and why?
- Compare [Entity_A], [Entity_B], [Entity_C], [Entity_D], regarding [Skill_Set]. Which player is more effective when operating under [Context] and why?
- Analyze [Entity]’s defensive performance during [Timeframe]. What type of opponent archetype does this player struggle against the most when defending [Spatial Information]?
- How did [Entity]’s performance in [Skill_Set] change before and after [Game Event] (e.g., substitution, tactical shift)?
- Analyze [Entity]’s offensive performance during [Timeframe]. What type of opponent archetype does this player struggle against the most when attacking [Spatial Information]?'''

prompt_causal_counterfactual_reasoning='''I want you to test students' ability to perform causal and counterfactual reasoning. Some example questions include:
- Analyze the sequence leading to [Negative Outcome] (e.g., goal, turnover, penalty) at [Temporal & Game State]. What was the primary root cause?
- What are the precise chain of events connecting the [Action A] at [Temporal & Game State] to the [Action B] at [Temporal & Game State]?
- Analyze the [Infractions & Penalties] (or potential non-call) involving [Entity] at [Temporal & Game State]. What happened and why?
- [Entity] executed [Strategic Metadata] (e.g. play, set piece) at [Temporal & Game State], but this time it failed. Diagnose why.
- What was the breakdown by [Opponent Entity] that led to the successful [Outcomes & Metrics] at [Temporal & Game State]?'''

prompt_anomaly_novelty_detection='''I want you to test students' ability to perform anomaly and novelty detection. Some potential templatesexample questions include:
- What is anomalous about the way [Entity] executed [Action Classification] at [Temporal & Game State]?
- What is unconventional about the lineup [Entity] employed at [Temporal & Game State]?
- What specific [Infractions & Penalties] (e.g., rule violation, foul type) was violated by [Entity] during [Context]?
- Did [Entity] deviate from their general game plan during [Timeframe]? If so, explain how and why.
- [Entity] typically relied heavily on [Skill_Set A]. Did they deviate from this tendency during [In-Game Period] and why?'''

prompt_spatiotemporal_relational_reasoning='''I want you to test students' ability to perform spatio-temporal and relational reasoning. Some example questions include:
- Which [Spatial Information] (e.g., zone name, the paint, the midfield, neutral zone) is [Entity] dominating or losing control of during [In-Game Period]?
- Identify the weak link in [Entity]’s defensive structure during [Timeframe].
- What was the critical action that occurred away from the primary focus (e.g., the ball/puck) on this play at [Temporal & Game State]? How did this action enable the final [Outcome]?
- Which [Spatial Information] (e.g., zone name, the paint, the midfield, neutral zone) did the numerical advantage or disadvantage occur during [Context]?
- Which [Spatial Information] (e.g., zone name, the paint, the midfield, neutral zone) does [Entity] occupy during [Strategic Metadata] (e.g., offensive play type, defensive scheme)?'''

prompt_general='''I want you to test students' ability to do complex reasoning, conduct deep analysis, and have an expert-level understanding of basketball. Some example questions include:
- What tactical adjustment did [Entity] make during [In-Game Period] (e.g., halftime, timeout)? Analyze the result.
- How did [Entity]’s performance in [Skill_Set] change before and after [Game Event] (e.g., substitution, tactical shift)?
- Analyze the [Infractions & Penalties] (or potential non-call) involving [Entity] at [Temporal & Game State]. What happened and why?
- Which [Spatial Information] (e.g., zone name, the paint, the midfield, neutral zone) is [Entity] dominating or losing control of during [In-Game Period]?
- What is unconventional about the lineup [Entity] employed at [Temporal & Game State]?'''

SYSTEM_PROMPT="""You will act as a teacher in a class called 'Sports Video Understanding.' Given a Question Category, team rosters, subtitles with timestamps, and game reports, your task is to generate difficult and diverse questions and corresponding answers for your students about the video, to later be used in a short answer setting.

The provided Question Category is not a strict format, and should instead be used as inspiration to generate questions of similar or more quality and depth.

**Guidelines and Restrictions:**
- Ensure each question does NOT give away its answer.
- Ensure each question and answer DO NOT contain any subtitles, timestamps, or quotes.
- Do NOT reference information that cannot be directly observed from watching the games themselves.
- Remember that students will only be provided with the game video without audio to answer the questions.
- The question and answers should NOT include any assumed details.
- The question should require more than simple action recognition or stats to answer.
- The question and answers MUST be in plain text, no formatting.
- The answers MUST be at most 50 words long.

**Output Requirements:**
Generate the five highest quality questions and corresponding answers based on the provided data. For each answer, provide evidence: timestamps for relevant subtitles and relevant quotes from the game report.
"""

PROMPT = """**Question Category:**
{0}

**Rosters:**
{1}

**Subtitles with timestamps:**
{2}

**Game reports:**
{3}
"""

base = """**Goal**
I want you to act as a teacher in a class called 'Sports Video Understanding.' Given team rosters, subtitles with timestamps, and game reports, generate difficult and diverse questions and corresponding answers for your students about the video to later be used in a short answer setting.

**Question Category**
{0}

These are not strict formats, and should instead be used as inspiration to generate questions of similar or more quality and depth.

**Rosters**
{1}

**Subtitles with timestamps**
{2}

**Game reports**
{3}

**Guidelines and Restrictions**
- Ensure each question does not give away its answer.
- Ensure each question and answer DO NOT contain any subtitles, timestamps, or quotes.
- Do NOT reference information that cannot be directly observed from watching the games themselves.
- Remember that students will only be provided with the game video without audio to answer the questions.
- The question and answers should NOT include any assumed details.
- The question should require more than simple action recognition or stats to answer.
- The question and answers MUST be in plain text, no formatting.
- The answers MUST be at most 50 words long.

Generate the five highest quality questions and corresponding answers. For each answer, provide evidence: timestamps for relevant subtitles and relevant quotes from the game report."""