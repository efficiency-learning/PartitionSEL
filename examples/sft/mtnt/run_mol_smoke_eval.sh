set -x # Enable xtrace

export WANDB_MODE="disabled"
export NCCL_DEBUG="NONE"

export JAX_LOG_COMPILES=0
export JAX_ENABLE_PGLE=false
export JAX_PLATFORMS=cuda,cpu
export PYTHONUNBUFFERED=1

NUM_NODES=1
NUM_GPUS=1
THRESHOLD_BYTES=1073741824
export XLA_FLAGS="\
--xla_gpu_autotune_level=4 \
--xla_gpu_deterministic_ops=false \
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

export JAX_DEFAULT_MATMUL_PRECISION="bfloat16"
export XLA_PYTHON_CLIENT_MEM_FRACTION=1
unset NUM_NODES NUM_GPUS THRESHOLD_BYTES

ROOT="/root/prayas/ret-subset/examples/sft/mtnt"
REPO_ROOT="/root/prayas/ret-subset"

export CUDA_VISIBLE_DEVICES=0
ORG="Qwen"
MODEL="Qwen2.5-0.5B"
MODEL_NAME="qwen2.5-0.5b"
EXPNAME="mol-qwen2.5-0.5b-smoke-1gpu-1step"
LORA='{"module_path":".*q_proj|.*k_proj|.*v_proj|.*gate_proj|.*down_proj|.*up_proj","rank":8,"alpha":16.0}'

EVAL_BENCHMARKS="mol"
EVAL_STEPS="1"
MOL_NUM_TEST="1"
MOL_BATCH_SIZE="1"
MOL_MAX_PROMPT_TOKENS="512"
MOL_MAX_GENERATION_STEPS="16"
export EVAL_BENCHMARKS EVAL_STEPS MOL_NUM_TEST MOL_BATCH_SIZE MOL_MAX_PROMPT_TOKENS MOL_MAX_GENERATION_STEPS

cd "$REPO_ROOT"
python3 -m tunix.cli.generate \
  base_config.yaml \
  model_config.model_name=$MODEL_NAME \
  model_config.model_id="$ORG/$MODEL" \
  model_config.model_source="huggingface" \
  model_config.rng_seed=0 \
  model_config.lora_config=$LORA \
  model_config.mesh.shape="(1,1)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  model_config.model_download_path="/tmp/models-ret-subset/$MODEL" \
  tokenizer_config.tokenizer_path="$ORG/$MODEL" \
  tokenizer_config.tokenizer_type="huggingface" \
  eval_batch_size=1 \
  max_target_length=512 \
  training_config.checkpoint_root_directory="$ROOT/ckpts/$EXPNAME" \