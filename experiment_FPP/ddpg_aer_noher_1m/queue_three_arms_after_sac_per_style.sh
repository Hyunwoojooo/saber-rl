#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
POLL_SECONDS="${POLL_SECONDS:-60}"

UPSTREAM_QUEUE_SESSION="${UPSTREAM_QUEUE_SESSION:-fpp_sac_per_style_after_td3}"
UPSTREAM_RUN_SESSION="${UPSTREAM_RUN_SESSION:-fpp_sac_per_style_1m}"
ORCHESTRATOR_SESSION="${ORCHESTRATOR_SESSION:-fpp_ddpg_aer_three_arm_after_sac}"

SAC_EXPERIMENT="$REPO_ROOT/experiment_FPP/sac_her_per_style_loss_weighting_ls512"
SAC_OUTPUT_ROOT="$REPO_ROOT/experiment_FPP/runs/sac_her_per_style_loss_weighting_ls512"
SAC_VALIDATOR="$SAC_EXPERIMENT/validate_results.py"

OUTPUT_ROOT="$REPO_ROOT/experiment_FPP/runs/fpp_ddpg_aer_noher_1m"
LOG_DIR="$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_noher_1m"
QUEUE_STATE_DIR="$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_three_arm_after_sac_queue"
QUEUE_LOG="$QUEUE_STATE_DIR/queue.log"
SOURCE_MANIFEST="$QUEUE_STATE_DIR/frozen_source.sha256"
LOCK_FILE="/tmp/saber-rl-fpp-ddpg-aer-three-arm-after-sac.lock"
CLAIM_DIR="$OUTPUT_ROOT/.three_arm_launch_claim"

SEED_LIST="1 2 3 4 5"
SEEDS=(1 2 3 4 5)
STEPS=1000000
ARMS=(ddpg_aer ddpg_er ddpg_annealing_er)

declare -A LAUNCHERS=(
  [ddpg_aer]="$SCRIPT_DIR/run_main_1m_ddpg_aer_seeds1_5_5parallel.sh"
  [ddpg_er]="$SCRIPT_DIR/run_main_1m_ddpg_er_seeds1_5_5parallel.sh"
  [ddpg_annealing_er]="$SCRIPT_DIR/run_main_1m_ddpg_annealing_er_seeds1_5_5parallel.sh"
)

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 2
fi
for required in tmux flock sha256sum find sort xargs pgrep tee; do
  if ! command -v "$required" >/dev/null 2>&1; then
    echo "Required command not found: $required" >&2
    exit 2
  fi
done
for arm in "${ARMS[@]}"; do
  if [[ ! -x "${LAUNCHERS[$arm]}" ]]; then
    echo "Launcher is not executable: ${LAUNCHERS[$arm]}" >&2
    exit 2
  fi
done
if [[ ! -x "$SAC_VALIDATOR" ]]; then
  echo "SAC validator is not executable: $SAC_VALIDATOR" >&2
  exit 2
fi
if [[ ! -x "$SCRIPT_DIR/validate_main_arm_gate.py" ]]; then
  echo "DDPG validator is not executable: $SCRIPT_DIR/validate_main_arm_gate.py" >&2
  exit 2
fi

mkdir -p "$QUEUE_STATE_DIR"
if [[ -e "$QUEUE_LOG" || -e "$SOURCE_MANIFEST" ]]; then
  echo "Refusing to overwrite existing queue state: $QUEUE_STATE_DIR" >&2
  exit 1
fi
exec > >(tee -a "$QUEUE_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

trap 'status=$?; log "orchestrator exit status=$status"' EXIT

exec 9>"$LOCK_FILE"
flock -n 9 || fail "another DDPG three-arm queue holds $LOCK_FILE"

if ! tmux has-session -t "$UPSTREAM_QUEUE_SESSION" 2>/dev/null \
  && ! tmux has-session -t "$UPSTREAM_RUN_SESSION" 2>/dev/null; then
  fail "neither upstream SAC queue nor SAC run session exists"
fi

for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="$OUTPUT_ROOT/fetch_pick_and_place_${arm}_noher_s${seed}_${STEPS}"
    arm_log="$LOG_DIR/${arm}_seed${seed}.log"
    [[ ! -e "$run_dir" ]] || fail "existing DDPG run blocks reservation: $run_dir"
    [[ ! -e "$arm_log" ]] || fail "existing DDPG log blocks reservation: $arm_log"
  done
done

mkdir -p "$OUTPUT_ROOT"
mkdir "$CLAIM_DIR" 2>/dev/null \
  || fail "DDPG three-arm launch already claimed: $CLAIM_DIR"
printf 'source_commit=a5436ef\nqueued_at=%s\nupstream_queue=%s\nupstream_run=%s\nsteps=%s\nseeds=%s\norder=%s\ndevice=%s\n' \
  "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" \
  "$UPSTREAM_QUEUE_SESSION" "$UPSTREAM_RUN_SESSION" \
  "$STEPS" "$SEED_LIST" "${ARMS[*]}" "$DEVICE" \
  >"$CLAIM_DIR/metadata.txt"
printf 'waiting_for_sac\n' >"$CLAIM_DIR/stage"

write_source_manifest() {
  (
    cd "$SCRIPT_DIR"
    find config.json run.py code/src \
      run_main_1m_ddpg_aer_seeds1_5_5parallel.sh \
      run_main_1m_ddpg_er_seeds1_5_5parallel.sh \
      run_main_1m_ddpg_annealing_er_seeds1_5_5parallel.sh \
      validate_main_arm_gate.py \
      queue_three_arms_after_sac_per_style.sh \
      -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) >"$SOURCE_MANIFEST"
}

verify_source_manifest() {
  (
    cd "$SCRIPT_DIR"
    sha256sum --check "$SOURCE_MANIFEST"
  )
}

write_source_manifest
log "queue armed; source manifest frozen at $SOURCE_MANIFEST"
log "waiting for SAC queue/run sessions: $UPSTREAM_QUEUE_SESSION or $UPSTREAM_RUN_SESSION"

while tmux has-session -t "$UPSTREAM_QUEUE_SESSION" 2>/dev/null \
  || tmux has-session -t "$UPSTREAM_RUN_SESSION" 2>/dev/null; do
  log "waiting for upstream SAC PER-style completion"
  sleep "$POLL_SECONDS"
done

log "upstream SAC sessions ended; checking for leftover SAC trainers"
SAC_TRAINER_PATTERN='sac_her_per_style_loss_weighting_ls512/code/src/train_fetch_pick_and_place_lite_v0.py'
if pgrep -af "$SAC_TRAINER_PATTERN" >/dev/null 2>&1; then
  pgrep -af "$SAC_TRAINER_PATTERN" || true
  fail "SAC trainer process remains after upstream sessions ended"
fi

log "validating all five completed SAC PER-style 1M runs"
"$PYTHON" "$SAC_VALIDATOR" \
  --output_root "$SAC_OUTPUT_ROOT" \
  --steps "$STEPS" \
  --seeds "${SEEDS[@]}"

validate_source_and_command() {
  local arm="$1"
  log "checking frozen source manifest before $arm"
  verify_source_manifest
  log "validating frozen command for $arm"
  "$PYTHON" "$SCRIPT_DIR/run.py" \
    --arm "$arm" \
    --seed 1 \
    --steps "$STEPS" \
    --device cpu \
    --validate_only
}

validate_arm_results() {
  local arm="$1"
  log "running strict completion gate for $arm seeds 1-5"
  "$PYTHON" "$SCRIPT_DIR/validate_main_arm_gate.py" \
    --arm "$arm" \
    --seeds "$SEED_LIST" \
    --output-root "$OUTPUT_ROOT" \
    --log-dir "$LOG_DIR"
}

run_arm() {
  local arm="$1"
  local launcher="${LAUNCHERS[$arm]}"
  validate_source_and_command "$arm"
  printf '%s\n' "$arm" >"$CLAIM_DIR/stage"
  log "starting $arm 1M seeds 1-5 in parallel"
  PYTHON="$PYTHON" \
  DEVICE="$DEVICE" \
  SEED_LIST="$SEED_LIST" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  LOG_DIR="$LOG_DIR" \
    bash "$launcher"
  log "$arm launcher completed; validating artifacts"
  validate_arm_results "$arm"
  log "$arm completion gate passed"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

log "SAC completion gate passed; beginning DDPG three-arm chain"
for arm in "${ARMS[@]}"; do
  run_arm "$arm"
done

printf 'complete\n' >"$CLAIM_DIR/stage"
log "DDPG three-arm chain completed and validated: ${ARMS[*]}"
