#!/usr/bin/env bash
# ===========================================================================
#  Run the quantization sweep from a folder, overriding whatever you edited.
#
#      ./qb.sh                                          defaults
#      ./qb.sh --levels int8_ao --n 1000 --gpus 16
#      ./qb.sh --levels int4_ao --n 20 --gpus 2,5,9 --mem-gb 8
#      ./qb.sh --levels bf16 --n 200 --ka 5 --ks 1 --kb 2     top-10
#      ./qb.sh --levels nf4 --ka 20 --ks 10 --kb 10 --purge-failed
#                                    delete that run's failed molecules so the
#                                    next run redoes them. No GPU, seconds.
#      ./qb.sh --levels nf4 --base fp16
#                                    compute NF4 in fp16 instead of bf16. Its
#                                    own file, .basefp16, so it never touches
#                                    a bf16 run.
#      ./qb.sh --dry-run             does THIS image have a path for each
#                                    level at all? No GPU, no model, ~1s.
#      ./qb.sh --dry-run --levels int2_ao,w2a8_intx      just these
#      ./qb.sh --native --levels int2_ao --ka 1 --ks 1 --kb 1 --greedy
#                                    no container: run the worker with the
#                                    host python. For a machine where the
#                                    image will not start but the GPU works.
#      ./qb.sh --probe               on a GPU: do the kernels run, and how
#                                    many BYTES PER WEIGHT does each level
#                                    actually hold? Minutes, no model load,
#                                    fits a laptop card.
#
#  The .sif already contains everything -- model, test set, chat template, all
#  scripts. Put an edited copy of any of those files next to it and this
#  launcher binds it over the baked-in one automatically. Delete the file and
#  the image's own version is used again. Nothing to remember, no bind list to
#  get wrong, and no rebuild to change a script.
#
#  Files it will pick up if present (host name -> path inside the image):
#      qb-run_llm.py | run_llm.py  -> /opt/qb/scripts/run_llm.py
#      run_llm_cluster.sh          -> /opt/qb/dist/run_llm_cluster.sh
#      score_sweep.py              -> /opt/qb/scripts/score_sweep.py
#      fidelity.py                 -> /opt/qb/scripts/fidelity.py
#      probe_capability.py         -> /opt/qb/scripts/probe_capability.py
#      chat_template.jinja         -> /opt/qb/data/chat_template.jinja
#      subset_full.jsonl           -> /opt/qb/data/subset_full.jsonl
#
#  Results go to <folder>/results, never into the folder itself, so the
#  distribution files stay clean and the folder can be copied elsewhere.
# ===========================================================================
set -u

# Printed on every run so a stale copy of this launcher identifies itself.
QB_LAUNCHER_REV="2026-09-27.1  (--ans-chunk passed through; unchanged otherwise)"

# The folder defaults to wherever this script lives, so you never type a path.
# A directory may still be given as the first argument for the case where the
# script and the image are not together.
if [ "$#" -ge 1 ] && [ -d "$1" ] && [ "${1#--}" = "$1" ]; then
  D="$(cd "$1" && pwd)"; shift
else
  D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# --native is read BEFORE the image search, because that search exits when it
# finds nothing and a native run needs neither an image nor singularity.
#
# Why this exists: on an RTX 3070 laptop under WSL the CUDA 13.3 image
# segfaults inside torch.cuda.is_available(). The GPU itself is fine -- it is
# the container that cannot drive it. run_llm_cluster.sh was already container-agnostic:
# QB_ROOT overrides /opt/qb, --model overrides the checkpoint path, and it
# discovers a python by testing candidates for `import torch, transformers`.
# So the whole harness runs outside the image with no new vocabulary.
NATIVE=0
for _a in "$@"; do [ "$_a" = "--native" ] && NATIVE=1; done

SIF=""; NSIF=0; SING=""
if [ "$NATIVE" = 0 ]; then
  # The image: any .sif in the folder. Named images are not special-cased, so a
  # rebuild under a new name is picked up with no edit.
  for cand in "$D"/*.sif; do
    [ -f "$cand" ] || continue
    NSIF=$((NSIF + 1)); SIF="$cand"
  done
  if [ "$NSIF" -eq 0 ]; then
    echo "no .sif found in $D  (use --native to run without a container)" >&2
    exit 1
  elif [ "$NSIF" -gt 1 ]; then
    echo "more than one .sif in $D -- keep only the one you want to run:" >&2
    ls -1 "$D"/*.sif >&2; exit 1
  fi

  SING="$(command -v singularity || command -v apptainer || true)"
  [ -n "$SING" ] || { echo "singularity/apptainer not installed" >&2; exit 1; }
fi

# Two options are handled HERE and not passed on to the runner.
#
#   --results DIR   score or resume a results directory that already exists
#                   somewhere else. Without this the only way to look at an
#                   earlier sweep was to copy its files into ./results, and a
#                   run against the wrong directory silently produced an empty
#                   table that read as "every level scored zero".
#   --score-only    re-score what is on disk and stop. No GPU, no model load,
#                   seconds rather than hours. This is what you want after
#                   changing score_sweep.py.
WORK=""; SCORE_ONLY=0; BENCH=0; BENCHCPU=0; PURGE=0; DRYRUN=0; PROBE=0; ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --results)    WORK="$2"; shift 2 ;;
    --score-only) SCORE_ONLY=1; shift ;;
    # Measure what each KV-cache strategy costs on THIS card: default (all on
    # GPU), offloaded to CPU RAM, and quantized int4. One beam search each, so
    # minutes rather than hours. Honours --levels and --kb.
    --bench-cache) BENCH=1; shift ;;
    # Is CPU-only viable, and does quantization speed a CPU up? Measures one
    # large linear layer per format -- seconds, and no GPU is touched.
    --bench-cpu)   BENCHCPU=1; shift ;;
    # Does THIS image expose the config class for each level at all? That is an
    # import and a constructor, not a kernel, so it needs no GPU, no model and
    # no allocation -- about a second, on a login node, while a sweep has the
    # card. Run it before queueing a level the image may not have. It cannot
    # tell you whether the kernels execute; a normal run answers that, since
    # run_llm_cluster.sh probes the card on its way in.
    --dry-run)     DRYRUN=1; shift ;;
    # Consumed here; the pre-scan above already acted on it.
    --native)      shift ;;
    # The same probe on a real GPU: every quantization path tried on an actual
    # matmul, plus bytes-per-weight measured from the allocator. It loads no
    # model, so it runs on a card far too small for the 8B weights -- which is
    # the point, because "does 2-bit save VRAM" is answerable on a laptop.
    # run_llm_cluster.sh already does this at the start of a real run; this
    # flag is for when you want the answer WITHOUT the run.
    --probe)       PROBE=1; shift ;;
    # Passed straight through to the runner, which owns the level-tag naming.
    # Noted here only so the GPU can be left out: deleting records needs none,
    # and a login node or a CPU-only allocation must be able to do it.
    --purge-failed) PURGE=1; ARGS+=("$1"); shift ;;
    *)            ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS+"${ARGS[@]}"}

[ -n "$WORK" ] || WORK="${QB_RESULTS:-$D/results}"
mkdir -p "$WORK" 2>/dev/null || true
if [ ! -d "$WORK" ]; then
  echo "results directory does not exist and could not be created: $WORK" >&2
  exit 1
fi
WORK="$(cd "$WORK" && pwd)"
BINDS=(--bind "$WORK:/work")

# Results written by an earlier `sudo` run are owned by root, and every later
# run as a normal user then fails on its first line with
#     tee: /work/logs/llm.<date>.log: Permission denied
# which looks like a container problem and is not one. Say so plainly.
if [ ! -w "$WORK" ]; then
  echo "cannot write to $WORK (owned by $(stat -c '%U' "$WORK" 2>/dev/null))" >&2
  echo "  a previous run under sudo left it root-owned. Fix with:" >&2
  echo "      sudo chown -R \"\$USER\" \"$WORK\"" >&2
  echo "  and do not use sudo -- this needs no privileges." >&2
  exit 1
fi

# WSL exposes the GPU through /dev/dxg with the driver libraries under
# /usr/lib/wsl, not through /dev/nvidia*. --nv passes /dev/dxg through by
# itself -- that part was never the problem -- but it knows nothing about
# /usr/lib/wsl, so the container gets the device node and no libcuda.so.1:
#     nvidia-smi:  Failed to initialize NVML: GPU access blocked by the OS
#     torch:       Found no NVIDIA driver on your system
# while the GPU works perfectly from the host. MEASURED on an RTX 3070 under
# WSL: --nv alone gives torch.cuda.is_available() False; adding the bind AND
# the linker path gives True. The bind alone is not enough -- the files become
# visible but the loader still cannot find them.
#
# LD_LIBRARY_PATH is set to exactly /usr/lib/wsl/lib and nothing else. Carrying
# the host's LD_LIBRARY_PATH in would point the container's loader at host
# paths that do not exist inside it.
#
# The variable is prefixed to match the runtime actually in use: apptainer
# warns that SINGULARITYENV_ is deprecated in its favour, and SingularityCE
# does not read APPTAINERENV_ at all. Setting both works but prints a notice
# on every single run.
if [ -d /usr/lib/wsl ] && [ "$NATIVE" = 0 ]; then
  BINDS+=(--bind /usr/lib/wsl:/usr/lib/wsl)
  if "$SING" --version 2>/dev/null | grep -qi apptainer; then
    export APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/wsl/lib"
  else
    export SINGULARITYENV_LD_LIBRARY_PATH="/usr/lib/wsl/lib"
  fi
  echo "  WSL detected: binding /usr/lib/wsl and putting its lib on the linker path"

  # NVIDIA's CUDA base images ship /usr/local/cuda*/compat/libcuda.so.*. That
  # directory exists to let a container use a NEWER driver than the host has,
  # on normal Linux. On WSL it is poison: the only libcuda that works is the
  # host stub in /usr/lib/wsl/lib, which talks to /dev/dxg, and the compat
  # build has no idea /dev/dxg exists.
  #
  # MEASURED 2026-09-16, RTX 3070 Laptop, host driver 610.88, image CUDA 13.3
  # shipping compat 610.43.02: torch.cuda.is_available() took a SEGFAULT with
  # both visible -- no Python traceback, nothing. The same laptop worked with
  # the older image, which is why the bind-plus-linker-path note above says
  # this combination was fine; the compat directory is the new variable, not
  # the driver and not the card.
  #
  # LD_LIBRARY_PATH alone does not settle it: ldconfig inside this image has NO
  # libcuda entry, so resolution falls through to RPATH/RUNPATH inside torch's
  # own shared objects, and those can reach compat regardless of what
  # LD_LIBRARY_PATH says. Masking the directory removes the candidate outright,
  # which is the only version of this that does not depend on load order.
  #
  # Real paths only: /usr/local/cuda and /usr/local/cuda-13 are symlinks to the
  # versioned directory, so readlink -f collapses all three to one bind.
  QB_EMPTY="$(mktemp -d 2>/dev/null || echo /tmp/qb-empty-$$)"
  mkdir -p "$QB_EMPTY" 2>/dev/null || true
  QB_COMPAT="$("$SING" exec "$SIF" sh -c \
      'for d in /usr/local/cuda*/compat; do [ -d "$d" ] && readlink -f "$d"; done' \
      2>/dev/null | sort -u)"
  if [ -n "$QB_COMPAT" ]; then
    for c in $QB_COMPAT; do BINDS+=(--bind "$QB_EMPTY:$c"); done
    echo "  masking CUDA compat libs so the WSL host driver is the only candidate:"
    for c in $QB_COMPAT; do echo "      $c"; done
  fi
fi

add() {   # add() <host filename> <path inside image>
  if [ -f "$D/$1" ]; then
    BINDS+=(--bind "$D/$1:$2")
    printf "  override: %-22s -> %s\n" "$1" "$2"
    return 0
  fi
  return 1
}

if [ "$NATIVE" = 1 ]; then
  # QB_ROOT needs scripts/ and data/ beside each other. From dist/ that is the
  # parent; from a repo checkout it is the folder itself. Anything else is a
  # flat unzip of the distribution, which has the scripts but NOT
  # data/subset_full.jsonl -- so say which piece is missing rather than failing
  # later on a path nobody typed.
  if [ -d "$D/scripts" ] && [ -d "$D/data" ]; then
    QB_ROOT_N="$D"
  elif [ -d "$D/../scripts" ] && [ -d "$D/../data" ]; then
    QB_ROOT_N="$(cd "$D/.." && pwd)"
  else
    echo "--native needs a checkout with scripts/ and data/ next to each other." >&2
    echo "  looked in: $D  and  $D/.." >&2
    echo "  Set QB_ROOT to such a folder and call run_llm_cluster.sh directly." >&2
    exit 1
  fi
  PYN=""
  for c in python3 python; do
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import torch' >/dev/null 2>&1; then PYN="$c"; break; fi
  done
  echo "launcher: $QB_LAUNCHER_REV"
  echo "mode    : --native  (no container)"
  echo "qb root : $QB_ROOT_N"
  echo "python  : ${PYN:-NONE FOUND WITH torch}"
  echo "results : $WORK"
  if [ -z "$PYN" ]; then
    echo "no python on PATH can import torch. Activate the environment first." >&2
    exit 1
  fi
  "$PYN" -c 'import torch;print("          torch",torch.__version__,"| cuda",
      torch.version.cuda,"| available",torch.cuda.is_available())' 2>&1 | tail -2
  echo
  if [ "$DRYRUN" = 1 ] || [ "$PROBE" = 1 ]; then
    mkdir -p "$WORK/probe" 2>/dev/null || true
    LV=""
    for i in $(seq 1 $#); do
      case "${!i}" in
        --levels) j=$((i+1)); [ "$j" -le "$#" ] && LV="${!j}" ;;
      esac
    done
    if [ "$DRYRUN" = 1 ]; then
      exec "$PYN" -u "$QB_ROOT_N/scripts/probe_capability.py" --dry-run \
        ${LV:+--levels="$LV"} "$WORK/probe/capability.dryrun.json"
    fi
    exec "$PYN" -u "$QB_ROOT_N/scripts/probe_capability.py" \
      "$WORK/probe/capability.json"
  fi
  if [ "$SCORE_ONLY" = 1 ]; then
    exec "$PYN" -u "$QB_ROOT_N/scripts/score_sweep.py" "$WORK/raw"
  fi
  [ "$#" -gt 0 ] || set -- --levels bf16 --ka 1 --ks 1 --kb 1 --greedy --gpus 1
  echo "args    : $*"
  echo
  exec env QB_ROOT="$QB_ROOT_N" QB_RESULTS="$WORK" \
    bash "$QB_ROOT_N/dist/run_llm_cluster.sh" "$@"
fi

echo "launcher: $QB_LAUNCHER_REV"
REAL="$(readlink -f "$SIF" 2>/dev/null || echo "$SIF")"
if [ "$REAL" != "$SIF" ]; then
  echo "image   : $SIF  ->  $REAL"
else
  echo "image   : $SIF"
fi
echo "          $(du -h "$REAL" 2>/dev/null | cut -f1)  $(date -r "$REAL" '+%Y-%m-%d %H:%M' 2>/dev/null)"
echo "results : $WORK"

# WHICH IMAGE IS THIS? Two .sif files look identical from outside, and running
# an image built for another GPU generation fails a different way in every level --
# "PackageNotFoundError: bitsandbytes" for the bitsandbytes arms, "no kernel
# image is available" for the torchao arms -- with the real cause buried in a
# UserWarning. One line, every run, so the question never has to be asked.
ID="$("$SING" exec "$SIF" cat /opt/qb/build-versions.json 2>/dev/null || true)"
if [ -n "$ID" ]; then
  echo "build   : $(printf '%s' "$ID" | tr -d '\n' | sed 's/[{}"]//g; s/  */ /g')"
else
  PYV="$("$SING" exec "$SIF" python -c 'import sys,torch;print(f"python {sys.version_info[0]}.{sys.version_info[1]}  torch {torch.__version__}")' 2>/dev/null || true)"
  [ -n "$PYV" ] && echo "build   : $PYV  (no build-versions.json -- older image)"
fi
# qb-run_llm.py and run_llm.py map to the same target; take the first that
# exists rather than binding twice, which Singularity rejects.
add qb-run_llm.py /opt/qb/scripts/run_llm.py || add run_llm.py /opt/qb/scripts/run_llm.py || true
add run_llm_cluster.sh   /opt/qb/dist/run_llm_cluster.sh   || true
add score_sweep.py       /opt/qb/scripts/score_sweep.py    || true
add fidelity.py          /opt/qb/scripts/fidelity.py       || true
add probe_capability.py  /opt/qb/scripts/probe_capability.py || true
add bench_cache.py       /opt/qb/scripts/bench_cache.py      || true
add bench_cpu.py         /opt/qb/scripts/bench_cpu.py        || true
add chat_template.jinja  /opt/qb/data/chat_template.jinja  || true
add subset_full.jsonl    /opt/qb/data/subset_full.jsonl    || true

if [ "$BENCHCPU" = 1 ]; then
  # No --nv: the point is to measure the machine WITHOUT the GPU. Passing --nv
  # would still work, but leaving it off proves the number owes nothing to the
  # card and lets this run while a sweep is using it.
  echo "args    : --bench-cpu  (no GPU used)"
  echo
  exec "$SING" exec "${BINDS[@]}" "$SIF" \
    python -u /opt/qb/scripts/bench_cpu.py
fi

if [ "$BENCH" = 1 ]; then
  # Reuse --levels and --kb rather than inventing new spellings: the point of
  # this launcher is that there is one vocabulary to remember.
  for i in $(seq 1 $#); do
    case "${!i}" in
      --levels) j=$((i+1)); export SINGULARITYENV_QB_LEVEL="${!j}"
                            export APPTAINERENV_QB_LEVEL="${!j}" ;;
      --kb)     j=$((i+1)); export SINGULARITYENV_QB_KB="${!j}"
                            export APPTAINERENV_QB_KB="${!j}" ;;
    esac
  done
  echo "args    : --bench-cache $*"
  echo
  exec "$SING" exec --nv "${BINDS[@]}" "$SIF" \
    python -u /opt/qb/scripts/bench_cache.py
fi

if [ "$DRYRUN" = 1 ]; then
  # Reuse --levels rather than inventing a new spelling: the point of this
  # launcher is that there is one vocabulary to remember.
  LV=""
  for i in $(seq 1 $#); do
    case "${!i}" in
      --levels) j=$((i+1)); [ "$j" -le "$#" ] && LV="${!j}" ;;
    esac
  done
  mkdir -p "$WORK/probe" 2>/dev/null || true
  echo "args    : --dry-run${LV:+ --levels $LV}  (no GPU, no model load)"
  echo
  exec "$SING" exec "${BINDS[@]}" "$SIF" \
    python -u /opt/qb/scripts/probe_capability.py --dry-run \
      ${LV:+--levels="$LV"} /work/probe/capability.dryrun.json
fi

if [ "$PROBE" = 1 ]; then
  mkdir -p "$WORK/probe" 2>/dev/null || true
  echo "args    : --probe  (GPU, no model load)"
  echo
  exec "$SING" exec --nv "${BINDS[@]}" "$SIF" \
    python -u /opt/qb/scripts/probe_capability.py /work/probe/capability.json
fi

if [ "$SCORE_ONLY" = 1 ]; then
  echo "args    : --score-only  (no GPU, no model load)"
  echo
  exec "$SING" exec "${BINDS[@]}" "$SIF" \
    python -u /opt/qb/scripts/score_sweep.py /work/raw
fi

[ "$#" -gt 0 ] || set -- --levels int8_ao --n 1000 --gpus 1
echo "args    : $*"
echo

NV=(--nv)
if [ "$PURGE" = 1 ]; then NV=(); echo "  (--purge-failed: no GPU requested)"; echo; fi

exec "$SING" exec ${NV+"${NV[@]}"} "${BINDS[@]}" "$SIF" \
  bash /opt/qb/dist/run_llm_cluster.sh "$@"
