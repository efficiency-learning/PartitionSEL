set -x # Enable xtrace

save_run_script() {
    local logdir="$1"
    local expname="$2"
    local script_path="$0"

    # If script is sourced, $0 is the shell, not the file.
    # Try to detect the real script path.
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

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export KAGGLE_USERNAME="${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
export KAGGLE_KEY="${KAGGLE_KEY:?set KAGGLE_KEY}"

export WANDB_MODE="disabled"
export NCCL_DEBUG="NONE"

export JAX_LOG_COMPILES=0
export JAX_ENABLE_PGLE=false
export JAX_PLATFORMS=cuda,cpu
export PYTHONUNBUFFERED=1

NUM_NODES=1
NUM_GPUS=1
THRESHOLD_BYTES=67108864
export XLA_FLAGS="\
--xla_gpu_autotune_level=2 \
--xla_gpu_deterministic_ops=false \
--xla_gpu_triton_gemm_any=false \
--xla_gpu_enable_latency_hiding_scheduler=false \
--xla_gpu_enable_pipelined_all_gather=false \
--xla_gpu_enable_pipelined_reduce_scatter=false \
--xla_gpu_enable_pipelined_all_reduce=false \
--xla_gpu_all_reduce_combine_threshold_bytes=$((THRESHOLD_BYTES/(NUM_NODES*NUM_GPUS))) \
--xla_gpu_all_gather_combine_threshold_bytes=$((THRESHOLD_BYTES/(NUM_NODES*NUM_GPUS))) \
--xla_gpu_reduce_scatter_combine_threshold_bytes=$((THRESHOLD_BYTES/(NUM_NODES*NUM_GPUS*2))) \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_command_buffer='' \
--xla_gpu_enable_highest_priority_async_stream=false \
--xla_gpu_enable_while_loop_double_buffering=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false"

export JAX_DEFAULT_MATMUL_PRECISION="bfloat16" # https://docs.jax.dev/en/latest/config_options.html#jax_default_matmul_precision

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
unset NUM_NODES NUM_GPUS THRESHOLD_BYTES
# unset XLA_FLAGS

ROOT="/root/prayas/ret-subset/examples/sft/mtnt"
REPO_ROOT="/root/prayas/ret-subset"
LOGDIR="$ROOT/results"

# EXPNAME="fg_mezov2-gauss_1e-3_greats-2e-5_4k"
# EXPNAME="fg_nrl_warm128-noproj-tanh-1,2_16_16_1e-4_lin_2k_greats-2e-5_4k"
# EXPNAME="fg_nrl_warm128-proj-fin-tanh-1,2_16_16_2e-4-rand0.5-act_lin_2k_greats-2e-5_4k"
# EXPNAME="fg-mol_nrlAll-jachidProjouterOnly_warm128_16_32_2e-4-adamw_lin_2k_greats-2e-5_4k-deveval"
# EXPNAME="fg-3b-mol-random-fastgrad-2e-5_4k-deveval"
# EXPNAME="fg-mol_mezo_warm128-L6_16_32_2e-4-adamw_lin_2k_norm-2e-5_4k-deveval"
# EXPNAME="fg-law-std_warm128-L6_16_16-probe2_2e-4-adamw_lin_2k_greats-2e-5_4k-deveval"
# EXPNAME="fg-meta-mezo_warm128-L12_16_16-probe2_2e-4-mask-TT-adamw_lin_2k_gradnorm-2e-5_4k-deveval"
# EXPNAME="fg-meta-random-2e-5_4k-deveval"
# EXPNAME="legal-greatsJointv2-noadam-apgd512-8k-bs8x8-0.5-lr2e-4"
# EXPNAME="math-qwen3-4b-greatsJointv2-2k-bs16x4-0.25"
# EXPNAME="llama3-mol12.5-greats-norm-nolp-2k-32x4-v2"
# EXPNAME="math-2e-5-iwd-norm-nolp-diagfix-1k-32x4-v2"
EXPNAME="llama3-mol12.5-greats-norm-nolp-2k-32x4-v2"
# EXPNAME="timingv2-joint"

# EXPNAME="fg-law-nrl_random-2e-5-lora16_4k-deveval"

# EXPNAME="fg-mol-random-mezo8-2e-5_4k-deveval"
# EXPNAME="fg-mol_mezoV2-common_greats-2e-5_4k-deveval"
# EXPNAME="test"
# EXPNAME="fg-mol-random-nrl-L8-equal-dim16-deveval"
# EXPNAME="fg-gradtest"
# EXPNAME="fg_random_2e-6_10k"
mkdir -p $LOGDIR/$EXPNAME
export JAX_TRACEBACK_FILTERING=off
export HF_DATASETS_OFFLINE=0  # if you want no network calls
export TRANSFORMERS_VERBOSITY=warning
export HF_DATASETS_LOG_LEVEL=warning
# unset JAX_TRACEBACK_FILTERING

save_run_script "$LOGDIR" "$EXPNAME"

export CUDA_VISIBLE_DEVICES=2
# ORG="Qwen"
# MODEL="Qwen2.5-3B"
# MODEL_NAME="qwen2.5-3b"
ORG="meta-llama"
MODEL="Llama-3.2-3B"
MODEL_NAME="llama3.2-3b"
# facloc -> colm
# greats -> greatsDomain
# gradnorm -> gradnorm

TASK_CFG="$ROOT/task_config.yaml"

LORA='{"module_path":".*q_proj|.*k_proj|.*v_proj|.*gate_proj|.*down_proj|.*up_proj","rank":16,"alpha":96.0}'
# LORA='{"module_path":".*gate_proj|.*down_proj|.*up_proj","rank":8,"alpha":16.0}'
# LORA='{}'
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
  batch_size=32 \
  eval_batch_size=8 \
  eval_split=0.005 \
  max_target_length=512 \
  optimizer_config.opt_type="adamw" \
  subset_select.enabled=true \
  subset_select.ratio=0.125 \
  subset_select.buffer=8 \
  subset_select.mode=greats \
  optimizer_config.learning_rate=2e-4 \
  optimizer_config.warmup_ratio=0.03 \
  training_config.eval_every_n_steps=128 \
  training_config.max_steps=1024 \
  training_config.max_inflight_computations=32 \
  training_config.gradient_accumulation_steps=1 \
  training_config.metrics_logging_options.log_dir="$ROOT/tensorboard/$EXPNAME" \
  training_config.metrics_logging_options.flush_every_n_steps=1 \
  training_config.checkpoint_root_directory="$ROOT/ckpts/$EXPNAME" \
  | tee $LOGDIR/$EXPNAME/logs.log \