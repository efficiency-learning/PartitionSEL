
set -x # Enable xtrace

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export KAGGLE_USERNAME="${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
export KAGGLE_KEY="${KAGGLE_KEY:?set KAGGLE_KEY}"

export WANDB_MODE="disabled"
export NCCL_DEBUG="NONE"

export JAX_LOG_COMPILES=0
export JAX_ENABLE_PGLE=false
export PYTHONUNBUFFERED=1

NUM_NODES=1
NUM_GPUS=2
THRESHOLD_BYTES=1073741824
export XLA_FLAGS="\
--xla_gpu_autotune_level=6 \
--xla_gpu_deterministic_ops=true \
--xla_gpu_triton_gemm_any=false \
--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_all_reduce_combine_threshold_bytes=$((THRESHOLD_BYTES/(NUM_NODES*NUM_GPUS))) \
--xla_gpu_all_gather_combine_threshold_bytes=$((THRESHOLD_BYTES/(NUM_NODES*NUM_GPUS))) \
--xla_gpu_reduce_scatter_combine_threshold_bytes=$((THRESHOLD_BYTES/(NUM_NODES*NUM_GPUS*2))) \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_disable_hlo_passes=rematerialization \
--xla_gpu_enable_command_buffer='' \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false"

export JAX_DEFAULT_MATMUL_PRECISION="bfloat16" # https://docs.jax.dev/en/latest/config_options.html#jax_default_matmul_precision

export XLA_PYTHON_CLIENT_MEM_FRACTION=1
unset NUM_NODES NUM_GPUS THRESHOLD_BYTES

# ORG="Qwen"
# MODEL="Qwen2.5-3B"
# MODEL_NAME="qwen2.5-3b"
ORG="meta-llama"
MODEL="Llama-3.2-3B"
MODEL_NAME="llama3.2-3b"
ROOT="/root/prayas/ret-subset/examples/sft/mtnt"

# EXPNAME="legal-greatsJointv2-noadam-apgd512-8k-bs8x8-0.5-lr2e-4"
# EXPNAME="math-qwen3-greatsJointv2-noadam-8k-bs8x8-0.5"
# EXPNAME="math-qwen3-jointv2-noadam-8k-bs8x8-0.5"
# EXPNAME="math-qwen3-jointv2-8k-bs8x8-0.5"
# EXPNAME="mol-random-2k-32x4-v2"
# EXPNAME="llama3-mol12.5-greats-norm-nolp-2k-32x4-v2"
# EXPNAME="math-2e-5-iwd-norm-nolp-diagfix-1k-32x4-v2"
EXPNAME="llama3-mol12.5-gradnorm-2k-32x4-v2"

EVAL_SPLIT="${1:-test}"  # "dev" or "test" or "all"; default: dev

export CUDA_VISIBLE_DEVICES=2
LORA='{"module_path":".*q_proj|.*k_proj|.*v_proj|.*gate_proj|.*down_proj|.*up_proj","rank":16,"alpha":96.0}'
# LORA='{}'
EVAL_BENCHMARKS="mol"  # comma-separated: colm,legalbench,mol
# EVAL_STEPS="1536,1664,1792,1920,2048"
# EVAL_STEPS="1536,2048"
EVAL_STEPS="896,1024"

export EVAL_SPLIT EVAL_STEPS EVAL_BENCHMARKS
python3 -m tunix.cli.generate \
  base_config.yaml \
  model_config.model_name=$MODEL_NAME \
  model_config.model_id="$ORG/$MODEL" \
  model_config.model_source="huggingface" \
  model_config.rng_seed=0 \
  model_config.lora_config=$LORA \
  model_config.mesh.shape="(1,1)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  model_config.model_download_path="/tmp/models3/$MODEL" \
  tokenizer_config.tokenizer_path="$ORG/$MODEL" \
  tokenizer_config.tokenizer_type="huggingface" \
  eval_batch_size=1 \
  max_target_length=512 \
  training_config.checkpoint_root_directory="$ROOT/ckpts/$EXPNAME" \
