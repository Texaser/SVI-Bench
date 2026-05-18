# Auto-detect T3_ROOT: walk up to find pyproject.toml, then <repo>/data/t3.
# Override with T3_ROOT env var if your data lives elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${T3_ROOT:-}" ]; then
    _dir="$SCRIPT_DIR"
    while [ "$_dir" != "/" ] && [ ! -f "$_dir/pyproject.toml" ]; do
        _dir="$(dirname "$_dir")"
    done
    if [ -d "$_dir/data/t3" ]; then
        T3_ROOT="$_dir/data/t3"
    else
        echo "ERROR: T3_ROOT not set and could not auto-detect <repo>/data/t3." >&2
        echo "       Set T3_ROOT explicitly to a directory containing data/, embeds/, ckpts/, compositions/." >&2
        exit 1
    fi
fi

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='finetune_full_caption'
OUTPUT_DIR="${T3_ROOT}/results/${JOB_NAME}"
PARTITION="${SLURM_PARTITION:-a6000}"
NNODE=1
NUM_GPUS=8
NUM_CPU=8

# Per-epoch evaluation is skipped by default to keep training fast.
# Set EVAL_DURING_TRAINING=1 to run val-set R@K after each epoch.
JUMP_EVALUATE=$([ "${EVAL_DURING_TRAINING:-0}" = "1" ] && echo False || echo True)

mkdir -p "${OUTPUT_DIR}"

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=${NUM_CPU} \
    bash torchrun.sh \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    tasks/pretrain.py \
    scripts/finetuning/1B/config_finetune_full_caption.py \
    output_dir ${OUTPUT_DIR} \
    evaluate False \
    jump_evaluate ${JUMP_EVALUATE} \
    pretrained_path "${T3_ROOT}/ckpts/InternVideo2-stage2_1b-224p-f4.pt" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"


