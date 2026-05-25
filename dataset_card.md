<!--
Template for the HuggingFace dataset repo's README.md ("dataset card").
Publish via: copy this file's content to README.md on
https://huggingface.co/datasets/MVP-Group/SVI-Bench
-->

---
license: cc-by-nc-4.0
gated: true
extra_gated_prompt: "Please describe your intended research use of this dataset."
extra_gated_fields:
  Name: text
  Affiliation: text
  Country: country
  Intended use:
    type: select
    options:
      - Research
      - Education
      - Commercial evaluation
      - label: Other
        value: other
  I agree to the dataset terms: checkbox
---

# SVI-Bench

Multi-task benchmark for sports video understanding.

- **Project page:** https://svi-bench.github.io/
- **Code:** https://github.com/Texaser/SVI-Bench

Nine tasks across four pillars (Perception, Reasoning, Simulation, Agency),
three sports (basketball, hockey, soccer). See the code repository for
evaluation scripts and per-task READMEs.

## Repository layout

```
T1/{basketball,hockey,soccer,captions}/        Structured play-by-play description
T2/{basketball,hockey,soccer,data}/            Fine-grained action QA
T3/{clips,ckpts,embeds,compositions,data}/     Compositional video retrieval
T4/{basketball,hockey,soccer}/                 Strategic reasoning QA
T5/{basketball,soccer}/                        Outcome forecasting
T6/soccer/                                     Long-form narrative synthesis
T7/{basketball,soccer}/                        Motion-conditioned generation
T7/tracker_weights/                            YOLOX + MixFormer-ViT (shared with T8)
T8/basketball/                                 Goal-conditioned action generation
T8/llava_qa_checkpoint/                        Fine-tuned LLaVA-Qwen QA model
T8/tracker_weights/                            YOLOX + MixFormer-ViT
T9/{data,ckpts,embeds,questions,storage}/      Cross-corpus agentic reasoning
```

### T7 / T8 generation tasks

Per sport, each clip carries three files:

```
clips/{bucket}/{ID}.mp4         5 s game clip, 832×480, 15 fps
bboxes/{bucket}/{ID}.txt        per-frame player bboxes
backgrounds/{bucket}/{ID}.mp4   player-removed background
```

Sample IDs (`{ID}`) are zero-padded integers; `{bucket}` is `ID // 1668`
(T7 basketball), `ID // 1236` (T7 soccer), or `ID // 741` (T8 basketball).
Buckets are shipped as `.tar` bundles to stay under HF per-folder limits.

T7 splits: `T7/{basketball,soccer}/splits/{train,val,test,test_100}.txt`.
T8 splits: `T8/basketball/splits/{train,val,test,test_{100,1000}}.txt`.

T8 also ships:

- `captions.json` — per-ID `{refined_instruction, player_specifications}`.
- `qa_test/Q*.json` — multi-choice QA bank used by goal-accuracy eval.

### Download

The code repo provides a one-liner that snapshots T7 + T8 and unpacks the
tar bundles:

```bash
git clone https://github.com/Texaser/SVI-Bench
cd SVI-Bench
pip install "svi-bench[t7,t8]"
bash scripts/download_t7_t8.sh
```

For other tasks, see each task's README in the code repo.

## License

CC BY-NC 4.0. Research and educational use only; no redistribution.

## Citation

See the code repository for the BibTeX entry.
