# Updates for T3

This directory is a slim subset of
[`OpenGVLab/InternVideo`](https://github.com/OpenGVLab/InternVideo)
(`InternVideo2/multi_modality/`, commit `4b0b701`) with the following
additions and modifications for SVI-Bench T3.

## Added

Under `scripts/finetuning/1B/`:

- 12 eval shell scripts + 12 eval configs (`eval_{full_caption,attribute_dropout}_{test,val}_{basketball,hockey,soccer}.sh` and matching `config_eval_*.py`)
- 2 finetune shell scripts + 2 finetune configs (`finetune_{full_caption,attribute_dropout}.sh` and matching `config_finetune_*.py`)

## Modified

- `configs/data.py` — sports retrieval corpus entries.
- `dataset/{__init__,base_dataset,ret_dataset}.py` — SVI test format.
- `models/{criterions,internvideo2_stage2}.py` — sports-specific heads/losses.
- `tasks/{pretrain,retrieval_utils,shared_utils}.py` — retrieval eval + embedding-saving hook.

## License

Upstream license preserved verbatim in [`LICENSE`](LICENSE).
