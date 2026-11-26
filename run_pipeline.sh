#!/bin/bash
# Submit the em-pipe steps for one dataset to Slurm, each job waiting for the previous one.
#
# Usage: ./run_pipeline.sh CONFIG [--steps a,b,...] [--from STEP] [--to STEP] [--after JOBID] [--dry-run]
#   steps (in order): check preview refmask stitch align shading overlaps patchgrid zrelations refstats intensity
#                     render pyramid export quality summary precomputed upload
#   --steps a,b  only these steps (the first one waits for nothing)
#   --from STEP  STEP and every later step
#   --to STEP    every step up to and including STEP (with --from: the steps between the two)
#   --after ID   the first job waits for Slurm job ID to finish successfully
#   --summary-after ID[:ID...]  the summary job also waits for these jobs (e.g. another chain's last job)
#   --dry-run    print the sbatch commands instead of submitting them
# refmask, shading, overlaps, patchgrid, refstats, precomputed and upload run only when their enabled
# setting is true; zrelations only when intensity.balance_z_distance > 0.
# quality: the per-slice quality flags (tools/slice_quality.py). summary: the QC page of the volume
# (tools/qc_summary.py), the last job: it waits for every job submitted before it, whether they succeed or
# not, so when a job fails (and the jobs after it are cancelled) the page says where the chain stopped and why.
# Array sizes, CPUs, memory, time, partition and gres come from the config's slurm: section (see
# configs/example.yaml; slurm.auto sizes them from the dataset); logs go to output_dir/work/logs/.
# The config is checked first (python -m pipeline.config --config CONFIG --validate): with an unknown key, settings
# that do not work together, a raw_dir without matching files or a missing EsB file, nothing is submitted.
# Set EM_PIPE_SKIP_ENV=1 to skip sourcing slurm/env.sh (when the environment is already active).
# render init exits 1 (cancelling render and pyramid) when its inputs changed since the volume was made,
# e.g. after new data: re-create it with  sbatch slurm/render.sbatch CONFIG init --overwrite  (deletes
# the old volume), then  ./run_pipeline.sh CONFIG --from render.
set -eo pipefail

STEPS="check preview refmask stitch align shading overlaps patchgrid zrelations refstats intensity render pyramid export quality summary precomputed upload"
usage() { sed -n '4,25p' "$0" | sed 's/^# \{0,1\}//' >&2; exit "$1"; }

CONFIG= SELECT= FROM= TO= AFTER= SUMMARY_AFTER= DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --steps) [ $# -ge 2 ] || usage 1; SELECT=$2; shift 2 ;;
    --from) [ $# -ge 2 ] || usage 1; FROM=$2; shift 2 ;;
    --to) [ $# -ge 2 ] || usage 1; TO=$2; shift 2 ;;
    --after) [ $# -ge 2 ] || usage 1; AFTER=$2; shift 2 ;;
    --summary-after) [ $# -ge 2 ] || usage 1; SUMMARY_AFTER=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage 1 ;;
    *) [ -z "$CONFIG" ] || usage 1; CONFIG=$1; shift ;;
  esac
done
[ -n "$CONFIG" ] || usage 1
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
[ -z "$SELECT" ] || [ -z "$FROM$TO" ] || { echo "use --steps or --from / --to, not both" >&2; exit 1; }
for s in ${SELECT//,/ } $FROM $TO; do
  [[ " $STEPS " == *" $s "* ]] || { echo "unknown step: $s (steps: $STEPS)" >&2; exit 1; }
done
# Jobs start in the repo root, so the config path must not be relative to here.
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

cd "$(dirname "$0")"
export EM_PIPE_ROOT=$PWD
[ "${EM_PIPE_SKIP_ENV:-0}" = 1 ] || source slurm/env.sh

cfg() { python -m pipeline.config --config "$CONFIG" "$@"; }

# Before anything is submitted: unknown keys, settings that do not work together, raw_dir, and with
# decurtaining from a reference detector the EsB file of every raw file (pipeline/preflight.py).
python -m pipeline.preflight "$CONFIG" --quiet || exit 1

position() {  # position STEP: its place in STEPS (1, 2, ...)
  local n=0 s
  for s in $STEPS; do n=$((n + 1)); [ "$s" != "$1" ] || { echo $n; return; }; done
}
FIRST=1 LAST=$(position upload)
[ -z "$FROM" ] || FIRST=$(position "$FROM")
[ -z "$TO" ] || LAST=$(position "$TO")
[ "$FIRST" -le "$LAST" ] || { echo "--from $FROM comes after --to $TO" >&2; exit 1; }

wanted() {  # wanted STEP: is STEP selected?
  if [ -n "$SELECT" ]; then
    [[ ",$SELECT," == *",$1,"* ]]
  else
    local n; n=$(position "$1")
    [ "$n" -ge "$FIRST" ] && [ "$n" -le "$LAST" ]
  fi
}

# Everything needed from the config in two calls (python start-up can be slow on shared filesystems).
VALUES=$(cfg --get output_dir render.num_scales precomputed.enabled upload.enabled shading.enabled \
         overlaps.enabled patchgrid.enabled refmask.enabled refstats.enabled intensity.balance_z_distance)
SBATCH=$(cfg --sbatch-args)   # one line per job: the job, then its sbatch options, tab-separated
{ read -r OUT; read -r NUM_SCALES; read -r PRECOMPUTED; read -r UPLOAD; read -r SHADING
  read -r OVERLAPS; read -r PATCHGRID; read -r REFMASK; read -r REFSTATS; read -r ZDIST; } <<< "$VALUES"

PREV=$AFTER
CHAIN=()      # "name<TAB>job id" of every job submitted here, in order: the summary job reads it
ALL=          # their ids, colon-separated
submit() {  # submit JOB NAME STEP [ARGS...]: sbatch slurm/STEP.sbatch after the previous job
  local job=$1 name=$2 step=$3 words
  shift 3
  local opts=(--parsable "--job-name=$name")
  while IFS=$'\t' read -r -a words; do
    [ "${words[0]}" != "$job" ] || opts+=("${words[@]:1}")
  done <<< "$SBATCH"
  if [ "$job" = qc_summary ]; then   # after every job of the chain, however it ended
    local wait=${ALL}${SUMMARY_AFTER:+:$SUMMARY_AFTER}
    [ -n "$ALL" ] || wait=${AFTER}${SUMMARY_AFTER:+:$SUMMARY_AFTER}
    wait=${wait#:}
    [ -z "$wait" ] || opts+=("--dependency=afterany:$wait")
  elif [ -n "$PREV" ]; then
    opts+=("--dependency=afterok:$PREV" --kill-on-invalid-dep=yes)
  fi
  local cmd=(sbatch "${opts[@]}" "slurm/$step.sbatch" "$CONFIG" "$@")
  local id
  if [ "$DRY" = 1 ]; then
    printf '%q ' "${cmd[@]}"
    echo
    id="<$name>"
  else
    id=$("${cmd[@]}")
    id=${id%%;*}   # --parsable prints "id" or "id;cluster"
    echo "$name: job $id"
  fi
  [ "$job" = qc_summary ] && return   # nothing waits for the summary
  PREV=$id
  CHAIN+=("$name"$'\t'"$id")
  ALL=${ALL:+$ALL:}$id
}

[ "$DRY" = 1 ] || mkdir -p "$OUT/work/logs"

if wanted check; then submit check em-check check; fi
if wanted preview; then
  submit preview em-preview-run preview run
  submit preview_merge em-preview-merge preview merge
fi
if wanted refmask; then
  if [ "$REFMASK" = true ]; then
    submit refmask em-refmask-run refmask run
    submit refmask_merge em-refmask-merge refmask merge
  else
    echo "refmask.enabled is false: refmask skipped" >&2
  fi
fi
if wanted stitch; then
  submit stitch em-stitch-run stitch run
  submit stitch_merge em-stitch-merge stitch merge
fi
if wanted align; then
  submit align em-align-run align run
  submit align_solve em-align-solve align solve
fi
if wanted shading; then
  if [ "$SHADING" = true ]; then
    submit shading em-shading shading
  else
    echo "shading.enabled is false: shading skipped" >&2
  fi
fi
if wanted overlaps; then
  if [ "$OVERLAPS" = true ]; then
    submit overlaps em-overlaps-run overlaps run
    submit overlaps_merge em-overlaps-merge overlaps merge
  else
    echo "overlaps.enabled is false: overlaps skipped" >&2
  fi
fi
if wanted patchgrid; then
  if [ "$PATCHGRID" = true ]; then
    submit patchgrid em-patchgrid patchgrid
  else
    echo "patchgrid.enabled is false: patchgrid skipped" >&2
  fi
fi
if wanted zrelations; then
  if [ "${ZDIST:-0}" -gt 0 ] 2>/dev/null; then
    submit zrelations em-zrelations-run zrelations run
    submit zrelations_merge em-zrelations-merge zrelations merge
  else
    echo "intensity.balance_z_distance is 0: zrelations skipped" >&2
  fi
fi
if wanted refstats; then
  if [ "$REFSTATS" = true ]; then
    submit refstats em-refstats-run refstats run
    submit refstats_merge em-refstats-merge refstats merge
  else
    echo "refstats.enabled is false: refstats skipped" >&2
  fi
fi
if wanted intensity; then submit intensity em-intensity intensity; fi
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
if wanted quality; then
  submit slice_quality em-quality-run slice_quality
  submit slice_quality_merge em-quality-merge slice_quality merge
fi
SUMMARY=
if wanted summary; then
  # The chain so far, for the page: which jobs there were, so it can say where a chain stopped.
  CHAIN_FILE="$OUT/work/logs/chain_$(date +%Y%m%d_%H%M%S)_$(basename "${CONFIG%.*}").tsv"
  if [ "$DRY" = 1 ]; then
    CHAIN_FILE="<chain file>"
  else
    printf '%s\n' "${CHAIN[@]}" > "$CHAIN_FILE"
  fi
  KEEP=$PREV
  submit qc_summary em-qc-summary qc_summary --chain "$CHAIN_FILE"
  PREV=$KEEP SUMMARY=1
fi
if wanted precomputed; then
  if [ "$PRECOMPUTED" = true ]; then
    submit precomputed_init em-precomputed-init precomputed init
    submit precomputed em-precomputed-run precomputed run
    submit precomputed_verify em-precomputed-verify precomputed verify
  else
    echo "precomputed.enabled is false: precomputed skipped" >&2
  fi
fi
if wanted upload; then
  if [ "$UPLOAD" = true ]; then
    submit upload em-upload-run upload run
    submit upload_finish em-upload-finish upload finish
  else
    echo "upload.enabled is false: upload skipped" >&2
  fi
fi
[ -n "$PREV$SUMMARY" ] || { echo "no steps selected" >&2; exit 1; }
[ "$DRY" = 1 ] || echo "logs: $OUT/work/logs/   (squeue --me to follow)"
