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

JOB_NAME='eval_full_caption_test_hockey'
OUTPUT_DIR="${T3_ROOT}/results/${JOB_NAME}"
# Override EVAL_CKPT to point at your own .pth (defaults to the bundled
# sports-full baseline). EVAL_SUFFIX to a short tag — embeddings are saved as
# embeds/embeds_test_hockey_${EVAL_SUFFIX}.pt (default: "full").
EVAL_CKPT="${EVAL_CKPT:-${T3_ROOT}/ckpts/internvideo2_1b_sports_full.pth}"
EVAL_SUFFIX="${EVAL_SUFFIX:-full}"

mkdir -p "${OUTPUT_DIR}"

PARTITION="${SLURM_PARTITION:-a6000}"
NNODE=1
NUM_GPUS=1
NUM_CPU=8

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
    scripts/finetuning/1B/config_eval_full_caption_test_hockey.py \
    output_dir ${OUTPUT_DIR} \
    evaluate True \
    pretrained_path "${EVAL_CKPT}" \
    evaluation.embed_dir "${T3_ROOT}/embeds/embeds_test_hockey_${EVAL_SUFFIX}.pt" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
