#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
POLL_SECONDS="${POLL_SECONDS:-60}"

DDPG_SESSION="${DDPG_SESSION:-fpp_ddpg_aer_three_arm_after_sac}"
DDPG_EXPERIMENT="$REPO_ROOT/experiment_FPP/ddpg_aer_noher_1m"
DDPG_OUTPUT_ROOT="$REPO_ROOT/experiment_FPP/runs/fpp_ddpg_aer_noher_1m"
DDPG_LOG_DIR="$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_noher_1m"
DDPG_QUEUE_STATE="$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_three_arm_after_sac_queue"
DDPG_QUEUE_LOG="$DDPG_QUEUE_STATE/queue.log"
DDPG_SOURCE_MANIFEST="$DDPG_QUEUE_STATE/frozen_source.sha256"
DDPG_CLAIM_STAGE="$DDPG_OUTPUT_ROOT/.three_arm_launch_claim/stage"
DDPG_VALIDATOR="$DDPG_EXPERIMENT/validate_main_arm_gate.py"

OUTPUT_ROOT="$REPO_ROOT/experiment_FPP/runs/fpp_sac_aer_noher_1m"
LOG_DIR="$REPO_ROOT/experiment_FPP/logs/fpp_sac_aer_noher_1m"
QUEUE_STATE_DIR="$REPO_ROOT/experiment_FPP/logs/fpp_sac_aer_main_after_ddpg_queue"
QUEUE_LOG="$QUEUE_STATE_DIR/queue.log"
SOURCE_MANIFEST="$QUEUE_STATE_DIR/frozen_source.sha256"
LOCK_FILE="/tmp/saber-rl-fpp-sac-aer-main-after-ddpg.lock"
CLAIM_DIR="$OUTPUT_ROOT/.aer_after_ddpg_launch_claim"

AER_LAUNCHER="$SCRIPT_DIR/run_main_1m_sac_aer_seeds1_5_5parallel.sh"
SAC_VALIDATOR="$SCRIPT_DIR/validate_main_arm_gate.py"
SEED_LIST="1 2 3 4 5"
SEEDS=(1 2 3 4 5)
STEPS=1000000
DDPG_ARMS=(ddpg_aer ddpg_er ddpg_annealing_er)

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 2
fi
for required in tmux flock sha256sum find sort xargs pgrep rg tail tee git; do
  if ! command -v "$required" >/dev/null 2>&1; then
    echo "Required command not found: $required" >&2
    exit 2
  fi
done
for required_file in "$AER_LAUNCHER" "$SAC_VALIDATOR" "$DDPG_VALIDATOR"; do
  if [[ ! -x "$required_file" ]]; then
    echo "Required executable not found: $required_file" >&2
    exit 2
  fi
done
if ! tmux has-session -t "=$DDPG_SESSION" 2>/dev/null; then
  echo "DDPG upstream session is not running: $DDPG_SESSION" >&2
  exit 1
fi
if ! git -C "$REPO_ROOT" merge-base --is-ancestor \
  10d2a34e91c67b2a0440742e9ab8dcf2e32f6d21 HEAD; then
  echo "current HEAD does not contain SAC AER source commit 10d2a34" >&2
  exit 1
fi

mkdir -p "$QUEUE_STATE_DIR"
if [[ -e "$QUEUE_LOG" || -e "$SOURCE_MANIFEST" ]]; then
  echo "Refusing to overwrite existing queue state: $QUEUE_STATE_DIR" >&2
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  run_dir="$OUTPUT_ROOT/fetch_pick_and_place_sac_aer_noher_s${seed}_${STEPS}"
  arm_log="$LOG_DIR/sac_aer_seed${seed}.log"
  [[ ! -e "$run_dir" ]] || {
    echo "Existing SAC-AER run blocks reservation: $run_dir" >&2
    exit 1
  }
  [[ ! -e "$arm_log" ]] || {
    echo "Existing SAC-AER log blocks reservation: $arm_log" >&2
    exit 1
  }
done

mkdir -p "$OUTPUT_ROOT"
mkdir "$CLAIM_DIR" 2>/dev/null || {
  echo "SAC-AER launch was already claimed: $CLAIM_DIR" >&2
  exit 1
}
printf 'source_commit=10d2a34\nqueued_at=%s\nupstream_session=%s\nsteps=%s\nseeds=%s\narm=sac_aer\ndevice=%s\n' \
  "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" \
  "$DDPG_SESSION" "$STEPS" "$SEED_LIST" "$DEVICE" \
  >"$CLAIM_DIR/metadata.txt"
printf 'waiting_for_ddpg\n' >"$CLAIM_DIR/stage"

write_source_manifest() {
  (
    cd "$SCRIPT_DIR"
    find config.json run.py code/src \
      run_main_1m_sac_aer_seeds1_5_5parallel.sh \
      queue_sac_aer_main_after_ddpg_three_arm_s1_5.sh \
      validate_main_arm_gate.py \
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
exec > >(tee -a "$QUEUE_LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

finish() {
  status=$?
  if [[ "$status" -ne 0 ]]; then
    printf 'failed:%s\n' "$status" >"$CLAIM_DIR/stage"
  fi
  log "SAC-AER-after-DDPG queue exit status=$status"
}
trap finish EXIT

exec 9>"$LOCK_FILE"
flock -n 9 || fail "another SAC-AER-after-DDPG queue holds $LOCK_FILE"

log "queue armed; source manifest frozen at $SOURCE_MANIFEST"
log "waiting for DDPG three-arm session: $DDPG_SESSION"
while tmux has-session -t "=$DDPG_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

log "DDPG session ended; checking for leftover trainers and launchers"
DDPG_TRAINER_PATTERN='ddpg_aer_noher_1m/code/src/train_fetch_pick_and_place_lite_v0.py'
DDPG_LAUNCHER_PATTERN='run_main_1m_ddpg_.*_seeds1_5_5parallel.sh'
if pgrep -af "$DDPG_TRAINER_PATTERN" >/dev/null 2>&1; then
  pgrep -af "$DDPG_TRAINER_PATTERN" || true
  fail "DDPG trainer remains after upstream session ended"
fi
if pgrep -af "$DDPG_LAUNCHER_PATTERN" >/dev/null 2>&1; then
  pgrep -af "$DDPG_LAUNCHER_PATTERN" || true
  fail "DDPG launcher remains after upstream session ended"
fi

[[ -s "$DDPG_CLAIM_STAGE" ]] || fail "missing DDPG claim stage: $DDPG_CLAIM_STAGE"
[[ "$(<"$DDPG_CLAIM_STAGE")" == "complete" ]] \
  || fail "DDPG claim stage is not complete: $(<"$DDPG_CLAIM_STAGE")"
[[ -s "$DDPG_QUEUE_LOG" ]] || fail "missing DDPG queue log: $DDPG_QUEUE_LOG"
last_ddpg_status="$(rg 'orchestrator exit status=' "$DDPG_QUEUE_LOG" | tail -n 1 || true)"
[[ "$last_ddpg_status" == *"orchestrator exit status=0" ]] \
  || fail "DDPG orchestrator did not exit successfully: ${last_ddpg_status:-missing status}"
rg -F -q 'DDPG three-arm chain completed and validated: ddpg_aer ddpg_er ddpg_annealing_er' \
  "$DDPG_QUEUE_LOG" \
  || fail "DDPG queue log lacks the validated three-arm completion marker"

log "checking the DDPG frozen source manifest"
[[ -s "$DDPG_SOURCE_MANIFEST" ]] \
  || fail "missing DDPG source manifest: $DDPG_SOURCE_MANIFEST"
(
  cd "$DDPG_EXPERIMENT"
  sha256sum --check "$DDPG_SOURCE_MANIFEST"
)

for arm in "${DDPG_ARMS[@]}"; do
  log "rerunning strict DDPG completion gate for $arm seeds 1-5"
  "$PYTHON" "$DDPG_VALIDATOR" \
    --arm "$arm" \
    --seeds "$SEED_LIST" \
    --output-root "$DDPG_OUTPUT_ROOT" \
    --log-dir "$DDPG_LOG_DIR"
done

log "DDPG gates passed; checking frozen SAC-AER source"
verify_source_manifest
log "running SAC-AER sampler and runner regressions"
"$PYTHON" -B "$SCRIPT_DIR/code/src/test_aer_sampling.py"
"$PYTHON" -B "$SCRIPT_DIR/code/src/test_sac_aer_runner.py"
log "validating the frozen SAC-AER 1M command"
"$PYTHON" "$SCRIPT_DIR/run.py" \
  --arm sac_aer \
  --seed 1 \
  --steps "$STEPS" \
  --device cpu \
  --validate_only

printf 'sac_aer\n' >"$CLAIM_DIR/stage"
log "all gates passed; starting SAC-AER 1M seeds 1-5 in parallel"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PYTHON="$PYTHON" \
DEVICE="$DEVICE" \
SEED_LIST="$SEED_LIST" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
LOG_DIR="$LOG_DIR" \
  bash "$AER_LAUNCHER"

log "SAC-AER launcher completed; running strict completion gate"
"$PYTHON" "$SAC_VALIDATOR" \
  --arm sac_aer \
  --seeds "$SEED_LIST" \
  --output-root "$OUTPUT_ROOT" \
  --log-dir "$LOG_DIR"

printf 'complete\n' >"$CLAIM_DIR/stage"
log "SAC-AER 1M seeds 1-5 completed and validated"
