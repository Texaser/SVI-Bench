<!--
This file is the template for the HuggingFace dataset repo's README.md
(the "dataset card"). When publishing data to https://huggingface.co/datasets/MVP-Group/SVI-Bench,
copy this into that repo's README.md. Do NOT serve it from this code repo at runtime.

The YAML header below configures gated access. HF surfaces the prompt and
fields as a form on the dataset page; users must agree before their HF token
unlocks downloads.
-->

---
license: other
license_name: svi-bench-research-use
license_link: LICENSE
extra_gated_prompt: >
  By requesting access to SVI-Bench, you agree to the following terms:
  1. You will not redistribute any portion of this dataset.
  2. You will use this dataset solely for non-commercial research purposes.
  3. You will cite the SVI-Bench paper in any publication using this data.
extra_gated_fields:
  Full Name: text
  Affiliation: text
  Email: text
  I agree to the above terms: checkbox
configs:
  - config_name: shared
    data_files: shared/*.parquet
  - config_name: t1_structured_play_description
    data_files: t1_structured_play_description/*.parquet
  - config_name: t2_fine_grained_action_qa
    data_files: t2_fine_grained_action_qa/*.parquet
  - config_name: t3_compositional_video_retrieval
    data_files: t3_compositional_video_retrieval/*.parquet
  - config_name: t4_strategic_reasoning_qa
    data_files: t4_strategic_reasoning_qa/*.parquet
  - config_name: t5_outcome_forecasting
    data_files: t5_outcome_forecasting/*.parquet
  - config_name: t6_long_form_narrative_synthesis
    data_files: t6_long_form_narrative_synthesis/*.parquet
  - config_name: t7_motion_conditioned_generation
    data_files: t7_motion_conditioned_generation/*.parquet
  - config_name: t8_goal_conditioned_action_generation
    data_files: t8_goal_conditioned_action_generation/*.parquet
  - config_name: t9_cross_corpus_agentic_reasoning
    data_files: t9_cross_corpus_agentic_reasoning/*.parquet
  - config_name: all
    data_files:
      - shared/*.parquet
      - "t*/*.parquet"
---

# SVI-Bench

Multi-task benchmark with per-task configs.

- **Project page:** https://svi-bench.github.io/
- **Code:** https://github.com/Texaser/SVI-Bench

See the code repository for evaluation scripts and baselines.
