#!/bin/bash
# Submit the em-pipe steps for one dataset to Slurm, each job waiting for the previous one.
#
# Usage: ./run_pipeline.sh CONFIG [--steps a,b,...] [--from STEP] [--dry-run]
#   steps (in order): check preview stitch align intensity zcorrect render pyramid export
#   --steps a,b  only these steps (the first one waits for nothing)
#   --from STEP  STEP and every later step
#   --dry-run    print the sbatch commands instead of submitting them
# zcorrect runs only when zcorrect.enabled is true. Array sizes, CPUs, memory, time, partition and
# gres come from the config's slurm: section (see configs/example.yaml); logs go to output_dir/work/logs/.
# Set EM_PIPE_SKIP_ENV=1 to skip sourcing slurm/env.sh (when the environment is already active).
# render init exits 1 (cancelling render and pyramid) when its inputs changed since the volume was made,
# e.g. after new data: re-create it with  sbatch slurm/render.sbatch CONFIG init --overwrite  (deletes
# the old volume), then  ./run_pipeline.sh CONFIG --from render.
set -eo pipefail

STEPS="check preview stitch align intensity zcorrect render pyramid export"
usage() { sed -n '4,14p' "$0" | sed 's/^# \{0,1\}//' >&2; exit "$1"; }

CONFIG= SELECT= FROM= DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --steps) [ $# -ge 2 ] || usage 1; SELECT=$2; shift 2 ;;
    --from) [ $# -ge 2 ] || usage 1; FROM=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage 1 ;;
    *) [ -z "$CONFIG" ] || usage 1; CONFIG=$1; shift ;;
  esac
done
[ -n "$CONFIG" ] || usage 1
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
[ -z "$SELECT" ] || [ -z "$FROM" ] || { echo "use --steps or --from, not both" >&2; exit 1; }
for s in ${SELECT//,/ } $FROM; do
  [[ " $STEPS " == *" $s "* ]] || { echo "unknown step: $s (steps: $STEPS)" >&2; exit 1; }
done
# Jobs start in the repo root, so the config path must not be relative to here.
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

cd "$(dirname "$0")"
export EM_PIPE_ROOT=$PWD
[ "${EM_PIPE_SKIP_ENV:-0}" = 1 ] || source slurm/env.sh

cfg() { python -m pipeline.config --config "$CONFIG" "$@"; }

wanted() {  # wanted STEP: is STEP selected?
  if [ -n "$SELECT" ]; then
    [[ ",$SELECT," == *",$1,"* ]]
  elif [ -n "$FROM" ]; then
    [[ " ${STEPS#*"$FROM"} " == *" $1 "* || $1 == "$FROM" ]]
  fi
}

# Everything needed from the config in two calls (python start-up can be slow on shared filesystems).
VALUES=$(cfg --get output_dir zcorrect.enabled render.num_scales)
SBATCH=$(cfg --sbatch-args)   # one line per job: the job, then its sbatch options, tab-separated
{ read -r OUT; read -r ZCORRECT; read -r NUM_SCALES; } <<< "$VALUES"

PREV=
submit() {  # submit JOB NAME STEP [ARGS...]: sbatch slurm/STEP.sbatch after the previous job
  local job=$1 name=$2 step=$3 words
  shift 3
  local opts=(--parsable "--job-name=$name")
  while IFS=$'\t' read -r -a words; do
    [ "${words[0]}" != "$job" ] || opts+=("${words[@]:1}")
  done <<< "$SBATCH"
  [ -z "$PREV" ] || opts+=("--dependency=afterok:$PREV" --kill-on-invalid-dep=yes)
  local cmd=(sbatch "${opts[@]}" "slurm/$step.sbatch" "$CONFIG" "$@")
  if [ "$DRY" = 1 ]; then
    printf '%q ' "${cmd[@]}"
    echo
    PREV="<$name>"
  else
    PREV=$("${cmd[@]}")
    PREV=${PREV%%;*}   # --parsable prints "id" or "id;cluster"
    echo "$name: job $PREV"
  fi
}

[ "$DRY" = 1 ] || mkdir -p "$OUT/work/logs"

if wanted check; then submit check em-check check; fi
if wanted preview; then
  submit preview em-preview-run preview run
  submit preview_merge em-preview-merge preview merge
fi
if wanted stitch; then
  submit stitch em-stitch-run stitch run
  submit stitch_merge em-stitch-merge stitch merge
fi
if wanted align; then
  submit align em-align-run align run
  submit align_solve em-align-solve align solve
fi
if wanted intensity; then submit intensity em-intensity intensity; fi
if wanted zcorrect; then
  if [ "$ZCORRECT" = true ]; then
    submit zcorrect em-zcorrect-run zcorrect run
    submit zcorrect_solve em-zcorrect-solve zcorrect solve
  else
    echo "zcorrect.enabled is false: zcorrect skipped" >&2
  fi
fi
if wanted render; then
  submit render_init em-render-init render init
  submit render em-render-run render run
fi
if wanted pyramid; then
  for ((s = 1; s < NUM_SCALES; s++)); do
    submit pyramid "em-pyramid-s$s" pyramid run --scale "$s"
  done
fi
if wanted export; then submit export em-export export; fi
[ -n "$PREV" ] || { echo "no steps selected" >&2; exit 1; }
[ "$DRY" = 1 ] || echo "logs: $OUT/work/logs/   (squeue --me to follow)"
