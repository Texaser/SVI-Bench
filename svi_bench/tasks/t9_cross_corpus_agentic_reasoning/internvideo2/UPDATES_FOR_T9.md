# Updates for T9

This directory is a slim subset of
[`OpenGVLab/InternVideo`](https://github.com/OpenGVLab/InternVideo)
(`InternVideo2/multi_modality/`, commit `4b0b701`), bundled with T9 for
encoding videos with the InternVideo2 video encoder.

T9 uses InternVideo2 only via `demo/utils.py:setup_internvideo2` — for
video embedding extraction in the `search_videos` tool. T9 does not run
the upstream training or retrieval pipelines.

## Kept (used by T9)

- `models/`, `dataset/`, `configs/`, `utils/`, `tasks/` — encoder loading.
- `demo/utils.py`, `demo/internvideo2_stage2_config.py` — entrypoint for
  `setup_internvideo2`.

## Compatibility patches

Same as T3 (shared maintenance burden):

- `models/backbones/bert/tokenization_bert.py` — `self.vocab` loaded
  before `super().__init__()` so newer transformers' parent class can
  call `get_vocab()` during init.
- `models/backbones/internvideo2/internvideo2.py` — `flash_attn` imports
  wrapped in try/except (the symbols are optional; T9's runtime configs
  leave `use_flash_attn` / `use_fused_mlp` / `use_fused_rmsnorm` False).
- `models/backbones/bert/xbert.py` — kept upstream verbatim;
  `transformers>=4.45,<4.50` pin avoids the symbol-relocation issues.
- `dataset/__init__.py` — dangling `qa_dataset` import removed (the file
  isn't bundled).

## Removed (not used by T9)

- Upstream `requirements.txt` and `INSTALL.md` — they would conflict with
  SVI-Bench's `[t9]` extras.
- `tasks/`'s S3-checkpoint-push code paths — were dependent on an
  internal storage layer.

## License

Upstream license preserved verbatim in [`LICENSE`](LICENSE).
