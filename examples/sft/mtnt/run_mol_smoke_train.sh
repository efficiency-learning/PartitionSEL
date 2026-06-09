set -x # Enable xtrace

save_run_script() {
    local logdir="$1"
    local expname="$2"
    local script_path="$0"

    if [[ "$script_path" == "bash" || "$script_path" == "-bash" ]]; then
        echo "Warning: Script is being sourced - cannot auto-save."
        return
    fi

    local save_dir="$logdir/run_scripts/$expname"
    mkdir -p "$save_dir"

    local ts
    ts=$(date +"%Y%m%d_%H%M%S")

    local script_name
    script_name=$(basename "$script_path")

    local save_path="$save_dir/${script_name%.sh}_$ts.sh"

    cp "$script_path" "$save_path"

    echo "Saved run script to: $save_path"
}

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
LOGDIR="$ROOT/results"
EXPNAME="mol-qwen2.5-0.5b-smoke-1gpu-1step"
mkdir -p $LOGDIR/$EXPNAME

export JAX_TRACEBACK_FILTERING=off
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_VERBOSITY=warning
export HF_DATASETS_LOG_LEVEL=warning

save_run_script "$LOGDIR" "$EXPNAME"

export CUDA_VISIBLE_DEVICES=0
ORG="Qwen"
MODEL="Qwen2.5-0.5B"
MODEL_NAME="qwen2.5-0.5b"
TASK_CFG="$ROOT/task_config.yaml"
LORA='{"module_path":".*q_proj|.*k_proj|.*v_proj|.*gate_proj|.*down_proj|.*up_proj","rank":8,"alpha":16.0}'

cd "$REPO_ROOT"
python3 -m tunix.cli.peft_main \
  base_config.yaml \
  model_config.model_name=$MODEL_NAME \
  model_config.model_id="$ORG/$MODEL" \
  model_config.model_source="huggingface" \
  model_config.from_scratch=false \
  model_config.rng_seed=0 \
  model_config.lora_config=$LORA \
  model_config.mesh.shape="(1,1)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  model_config.model_download_path="/tmp/models-ret-subset/$MODEL" \
  tokenizer_config.tokenizer_path="$ORG/$MODEL" \
  tokenizer_config.tokenizer_type="huggingface" \
  dataset_name="zjunlp/Mol-Instructions" \
  task_config.task=ift \
  task_config.config="$TASK_CFG" \
  batch_size=1 \
  eval_batch_size=1 \
  eval_split=0.005 \
  max_target_length=512 \
  optimizer_config.opt_type="adamw" \
  subset_select.enabled=false \
  subset_select.ratio=0.25 \
  subset_select.buffer=4 \
  subset_select.mode=joint \
  optimizer_config.learning_rate=2e-5 \
  optimizer_config.warmup_ratio=0.03 \
  training_config.eval_every_n_steps=100000 \
  training_config.max_steps=1 \
  training_config.max_inflight_computations=1 \
  training_config.gradient_accumulation_steps=1 \
  training_config.metrics_logging_options.log_dir="$ROOT/tensorboard/$EXPNAME" \
  training_config.metrics_logging_options.flush_every_n_steps=1 \
  training_config.checkpoint_root_directory="$ROOT/ckpts/$EXPNAME" \
  | tee $LOGDIR/$EXPNAME/logs.log \