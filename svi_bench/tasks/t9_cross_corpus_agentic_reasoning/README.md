# T9 — Cross-Corpus Agentic Reasoning

T9 is the cross-corpus agentic reasoning task in SVI-Bench: an agent searches video clips and game documents (ESPN reports, game and season statistics) to answer sports questions, and a language-model judge grades the answer against ground truth.

## Install

```bash
conda create -n svi-bench-t9 python=3.11 -y && conda activate svi-bench-t9
pip install -e ".[t9]"
export HF_HOME=/path/to/hf_cache   # edit this for the hf cache location
python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-large-uncased')"
```

Download Elasticsearch 9.2.3 from elastic.co and disable security for local dev:

```bash
echo 'xpack.security.enabled: false' >> elasticsearch-9.2.3/config/elasticsearch.yml
```

## Data

```bash
huggingface-cli download MVP-Group/SVI-Bench --repo-type dataset \
    --include "T9/**" --local-dir data/
```

Large data (game archives, embeddings, ES indices) are shipped as `.tar`
bundles. After downloading, extract them:

```bash
python3 scripts/extract_tars.py --root data/T9
```

The data lives under `data/T9/`:

```
data/T9/
├── questions/
│   ├── basketball.json
│   ├── hockey.json
│   └── soccer.json
├── data/
│   ├── basketball/
│   ├── hockey/
│   └── soccer/
├── embeds/videos/
├── storage/                    # Elasticsearch index
└── ckpts/
    ├── internvideo2_sports.pth                       # video search encoder
    └── llava_next_video_sports_100k_f16_full_ft_hf/  # video_qa tool model
```

## Launch the environment

Start Elasticsearch and ingest the data (one-time, shared by both modes below):

```bash
elasticsearch-9.2.3/bin/elasticsearch -d
python3 scripts/ingest.py
curl http://localhost:9200/_cat/indices
```

Ingestion populates the document and video search indices. This runs once;
subsequent starts of `run_agent.py` / `run_batch.py` will detect the
populated indices and skip re-ingestion.

Then bring up the tool and orchestrator services.

### SLURM

```bash
bash scripts/submit_services.sh <arch-id> --node <hostname>
```

For `qwen3_235b` and `minimax_m2_5`, the tool services and the orchestrator run on separate nodes:

```bash
bash scripts/submit_services.sh <arch-id>_tools --node <tools-host>
bash scripts/submit_services.sh <arch-id> --node <orchestrator-host>
```

### Non-SLURM

```bash
python scripts/start_services.py --arch <arch-id>
```

For `qwen3_235b` and `minimax_m2_5`, the tool services and the orchestrator run on separate hosts:

```bash
# on the tools host
python scripts/start_services.py --arch <arch-id>_tools

# on the orchestrator host
python scripts/start_services.py --arch <arch-id>
```

## Evaluate

### Run the agent

```bash
export OPENAI_API_KEY=sk-...
export T9_ROOT=/path/to/data/T9                                       # default: <repo>/data/T9
export T9_ES_URL=http://<es-host>:9200                                # default: http://localhost:9200
export T9_TOOL_SERVER_HOST=<tools-host>                               # default: localhost
export T9_AGENT_SERVER_HOST=<orchestrator-host>                       # default: localhost
export CONDA_PROFILE=/path/to/miniconda3/etc/profile.d/conda.sh       # required for SLURM submission

bash scripts/submit_experiment.sh <arch-id> \
    --questions-file data/T9/questions/<sport>.json \
    --sport <sport>
```

### Run judges

```bash
svi-bench evaluate --task t9 --model <arch-id>
```

Looks at every completed run you have for that arch, asks which to score, and runs the OpenAI Batch judge on what you pick.

### Model cards

Oracle mode uses ground-truth event captions instead of raw videos for search and QA.

In the paper, we use A6000 (48GB) GPUs for all archs, and H100 (80GB) for the MiniMax-M2.5 agent host.

| `<arch-id>` | Model | GPUs |
|---|---|---|
| `gpt5` | OpenAI `gpt-5.2` | 5 (tools only; agent runs via API) |
| `gpt5_oracle` | OpenAI `gpt-5.2` | 5 (tools only; agent runs via API) |
| `qwen3_32b` | `Qwen/Qwen3-30B-A3B` | 7 |
| `qwen3_32b_oracle` | `Qwen/Qwen3-30B-A3B` | 7 |
| `qwen3_omni_30b` | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | 7 |
| `qwen3_omni_30b_oracle` | `Qwen/Qwen3-Omni-30B-A3B-Thinking` | 7 |
| `qwen3_235b` | `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8` | 8 (agent host) + 5 (tools host) |
| `qwen3_235b_oracle` | `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8` | 8 (agent host) + 5 (tools host) |
| `minimax_m2_5` | `MiniMaxAI/MiniMax-M2.5` | 4 (agent host) + 5 (tools host) |
| `minimax_m2_5_oracle` | `MiniMaxAI/MiniMax-M2.5` | 4 (agent host) + 5 (tools host) |

## Interactive mode

After the environment is up (see [Launch the environment](#launch-the-environment)):

```bash
python run_agent.py --arch <arch-id> --sport <sport>
```
