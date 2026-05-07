<!--
This file is the template for the HuggingFace dataset repo's README.md
(the "dataset card"). When publishing data to https://huggingface.co/datasets/svi-bench/svi-bench,
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
  - config_name: t1_scene_recognition
    data_files: t1_scene_recognition/*.parquet
  - config_name: t2_placeholder
    data_files: t2_placeholder/*.parquet
  - config_name: t3_action_recognition
    data_files: t3_action_recognition/*.parquet
  - config_name: t4_placeholder
    data_files: t4_placeholder/*.parquet
  - config_name: t5_placeholder
    data_files: t5_placeholder/*.parquet
  - config_name: t6_placeholder
    data_files: t6_placeholder/*.parquet
  - config_name: t7_deep_game_analysis
    data_files: t7_deep_game_analysis/*.parquet
  - config_name: t8_placeholder
    data_files: t8_placeholder/*.parquet
  - config_name: t9_placeholder
    data_files: t9_placeholder/*.parquet
  - config_name: all
    data_files:
      - shared/*.parquet
      - "t*/*.parquet"
---

# SVI-Bench

Multi-task benchmark with per-task configs. See the
[code repository](https://github.com/your-org/svi-bench) for evaluation
scripts and baselines.
