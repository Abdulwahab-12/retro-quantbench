#!/usr/bin/env bash
# ===========================================================================
#  Run RetroDFM-R-8B over the USPTO-50K test split, one worker per GPU.
#
#      singularity exec --nv --bind <results dir>:/work <image>.sif \
#          bash /opt/qb/dist/run_llm_cluster.sh --levels nf4 --ka 20 --ks 10 --kb 10
#
#  Normally started through qb.sh, which finds the image and does the
#  binding. Molecules are split across GPUs in interleaved shards and merged
#  into raw/ at the end. Resumable: finished molecules are read back and
#  skipped.
#
#  Options:
#     --gpus N|a,b,c  worker count, or physical device ids (default: all visible)
#     --n N           molecules                  (default 5005)
#     --levels a,b    quantization levels, comma or space separated
#                     (default: probe-driven -- fp16 plus whatever clears)
#     --ka N --ks N --kb N   inference augmentation (default 1 1 1)
#                     The augmented setting is --ka 20 --ks 10 --kb 10.
#                     The authors' eval script takes them as 10 10 20, which is
#                     (k_s, k_b, k_a) -- a different order. k_b is a count of
#                     answers sampled at temperature 1.4, not a beam width.
#     --mem-gb N      pretend the card has only N GB
#     --mem-phase X   all (default) caps loading too | infer caps only inference
#     --offload       spill layers that do not fit into host RAM (much slower)
#     --gen-chunk N   generate the k_s paths N at a time instead of all at once.
#                     Cuts peak KV cache ~k_s/N. For cards under 16 GB.
#     --ans-chunk N   generate the k_b answers N at a time. Lowers the peak of
#                     the answer stage; statistically the same, not bit-identical.
#     --model DIR     weights (default $QB_MODEL_DIR or /opt/model/retrodfm-r-8b)
#     --dry-run       print the plan and stop
# ===========================================================================
set -Eeuo pipefail

# Printed in the banner. Bump on every change that alters behaviour. This
# exists because an override file left beside the .sif is bound in preference
# to the image's own copy, and a STALE override is invisible: the run looks
# normal and behaves like an older version. Two debugging rounds were spent on
# "the fix did not work" when the fix simply was not the file being executed.
QB_RUNNER_REV="2026-09-27.2  (--ans-chunk; driver-level peak memory; ans_chunk in meta)"

# --------------------------------------------------------------------------- #
#  `set -e` + `set -o pipefail` + `cmd | head -N` is a trap, and it killed the
#  first cluster run at line 14 with no error message:
#
#      nvidia-smi --query-gpu=... | head -4
#
#  head exits after 4 lines, nvidia-smi is still writing the other 12, gets
#  SIGPIPE, and pipefail reports the pipeline as failed -- so set -e ends the
#  script. It works with 4 GPUs and dies with 16.
#
#  Likewise `[ test ] && cmd` as a standalone statement returns 1 when the test
#  is false, which set -e also treats as fatal. Both patterns are replaced with
#  explicit `if` blocks or guarded with `|| true` throughout this file.
# --------------------------------------------------------------------------- #

# Where the code and data live. Inside the Singularity image used for the
# reported runs this is /opt/qb; from a git clone it is the parent of the
# directory holding this script, so the repository works unmodified. Override
# with QB_ROOT if your layout differs.
QB="${QB_ROOT:-}"
if [ -z "$QB" ]; then
  QB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  [ -d "$QB/scripts" ] || QB=/opt/qb
fi

# Working directory: outputs, logs and the chat template. /work inside the
# image, ./out from a clone.
WORK="${QB_HOME:-}"
if [ -z "$WORK" ]; then
  if [ -d /work ]; then WORK=/work; else WORK="$PWD/out"; fi
fi

# No sensible default outside the image -- the weights are 16 GB and are not
# redistributed. Set QB_MODEL_DIR to wherever you downloaded them.
MODEL_DIR="${QB_MODEL_DIR:-/opt/model/retrodfm-r-8b}"
# The existence check used to sit HERE -- eleven lines above the loop that
# parses --model. So `--model /somewhere/real` exited with
# "model directory not found: /opt/model/retrodfm-r-8b", naming the default it
# had never been given a chance to replace. Inside the image the default always
# existed, which is why it survived. The check now lives after the loop.
N_FULL=5005; GPUS=""; LEVELS=""; DRY=0; PURGE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)    if [ "${2#*,}" != "$2" ]; then QB_GPU_LIST="$2";
               else GPUS="$2"; fi; shift 2 ;;
    --ka)      QB_KA="$2";  shift 2 ;;
    --ks)      QB_KS="$2";  shift 2 ;;
    --kb)      QB_KB="$2";  shift 2 ;;
    --mem-gb)  QB_MEM_LIMIT_GB="$2"; shift 2 ;;
    # all (default) caps loading too; infer caps only once the model is
    # resident, which is the "could this card RUN it" question rather than
    # "could this card BUILD it".
    --mem-phase) QB_MEM_LIMIT_PHASE="$2"; shift 2 ;;
    # Spill to host RAM instead of failing when the budget is exceeded.
    # accelerate places whole layers; the overflow streams over PCIe.
    --offload) QB_OFFLOAD=1; shift ;;
    # Deterministic decoding, both stages. Required for the un-augmented
    # 1x1x1 baseline: without it k_b=1 is ONE SAMPLE at T=1.4, not the argmax,
    # and cannot be compared with the paper's Table 1a 59.0%.
    --greedy)  QB_GREEDY=1; shift ;;
    # Compute dtype for the bitsandbytes levels (base dtype for torchao).
    # bf16 is the default and what every reported run used. fp16 is an
    # alternative for cards without native bfloat16. It goes into the
    # FILENAME (.basefp16) as well as the record
    # fingerprint -- without the tag, an fp16 run would share a file with the
    # bf16 run, fail the fingerprint check, and move days of it aside as
    # .stale, which is exactly what the aug/ans/alpha tags were added to stop.
    --base)    QB_BASE_DTYPE="$2"; shift 2 ;;
    # Generate the k_s sampled paths in groups of N instead of all at once.
    # Cuts peak KV cache roughly by k_s/N. Use on cards under ~16 GB when k_s
    # is large; leave off to reproduce the published runs bit-for-bit.
    --gen-chunk) QB_GEN_CHUNK="$2"; shift 2 ;;
    # --ans-chunk N   generate the k_b ANSWER samples N at a time. This is the
    #                 one that matters on a small card: the answer stage
    #                 prefills k_b copies of the whole reasoning prefix in ONE
    #                 allocation, and --gen-chunk does not touch it.
    --ans-chunk) QB_ANS_CHUNK="$2"; shift 2 ;;
    --n)       N_FULL="$2"; shift 2 ;;
    --levels)  LEVELS="$2"; shift 2 ;;
    --model)   MODEL_DIR="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    # Delete the failed molecules of THIS level and stop. A molecule with no
    # record is work to do, so the next run recomputes exactly those. Takes the
    # same --levels/--ka/--ks/--kb you would use to run it, because the files
    # it has to touch are named from them -- there is no separate tag to type.
    --purge-failed) PURGE=1; shift ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

# Levels may be given comma- or space-separated. The loop that consumes them
# word-splits on whitespace, so "bf16,int8_ao" arrived as ONE level named
# "bf16,int8_ao" and died inside a worker with a KeyError in a per-shard log --
# after the preflight had already passed, which is the worst place to find out.
LEVELS="${LEVELS//,/ }"

# Reject an unknown level HERE rather than in sixteen workers at once. A typo
# ("int8" for "int8_ao") otherwise costs a full launch cycle to discover.
# This list is a SECOND copy of run_llm.py's AO_SPECS keys plus its BNB keys,
# and the two drifted: int2_ao and w2a8_intx were added to the worker on
# 2026-09-14 and rejected here, so the level died at the launcher with
# "unknown level" and no hint that the worker supported it perfectly well.
# Keep them in step -- `qb.sh --dry-run` reads the worker's table directly and
# will list a level this whitelist refuses.
KNOWN="fp16 bf16 int8_ao w8a8 int4_ao w4a8 w4a4 \
w4a8_intx fp8_ao fp8_w8a8 fp8_w4a8 nvfp4 nvfp4_w mxfp8 mxfp4 \
int2_ao w2a8_intx \
nf4 nf4dq fp4"
for LV in $LEVELS; do
  case " $KNOWN " in
    *" $LV "*) ;;
    *) echo "unknown level: $LV"; echo "known levels: $KNOWN"; exit 2 ;;
  esac
done

mkdir -p "$WORK"/{raw,logs,probe,shards}
LOG="$WORK/logs/llm.$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m   ! %s\033[0m\n' "$*"; }

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# --------------------------------------------------------------------------- #
#  ENVIRONMENT RESOLUTION -- the block that made run_sweep.sh work in the image,
#  and which this script never got.
#
#  This script called bare `python`. Inside Singularity that is whatever is
#  first on PATH, which is frequently /usr/bin/python3 with no torch. Every
#  worker then dies instantly with ModuleNotFoundError into its own per-shard
#  log file, `wait` collects the failures, and the only thing visible on the
#  terminal is a run that "stopped".
#
#  Three rules, same as run_sweep.sh:
#    1. load the conda hook if conda exists at all
#    2. activate an environment ONLY if it really exists (QB_FORCE_ENV wins);
#       a missing env is not an error, it means the image needs no env
#    3. then pick an interpreter that can ACTUALLY import torch and
#       transformers, and use that absolute path everywhere afterwards
#
#  Rule 3 is the one that matters: asking the interpreter is the only way to
#  know. Inferring it from PATH is what cost the last four attempts.
# --------------------------------------------------------------------------- #
# CAPTURE THE CALLER'S INTERPRETER BEFORE THE CONDA HOOK RUNS.
#
# `eval "$(conda shell.bash hook)"` re-initialises conda in this shell and
# resets CONDA_PREFIX and PATH to the BASE installation. An environment the
# caller activated before invoking us is therefore discarded a few lines before
# anything looks for a python -- silently, because the hook succeeds.
#
# MEASURED 2026-09-16: prompt showing (syntheseus-quant), torch 2.8.0+cu128 and
# transformers 4.57.6 installed in it, qb.sh reporting that torch from the very
# same shell -- and this script then failing with "No module named 'torch'"
# from ~/miniconda3/bin/python, the BASE interpreter. Adding
# $CONDA_PREFIX to the search did not help: by the time the search ran,
# CONDA_PREFIX itself had been rewritten to base.
#
# So resolve first, ask questions later. Whatever the caller had active wins.
PRE_PY=""
for _c in ${CONDA_PREFIX:+"$CONDA_PREFIX/bin/python"} \
          ${VIRTUAL_ENV:+"$VIRTUAL_ENV/bin/python"} python3 python; do
  _r="$(command -v "$_c" 2>/dev/null || echo "$_c")"
  [ -x "$_r" ] || continue
  if "$_r" -c 'import torch, transformers' >/dev/null 2>&1; then
    PRE_PY="$_r"; break
  fi
done

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)" || true
  WANT="${QB_FORCE_ENV:-}"
  if [ -n "$WANT" ] \
     && conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$WANT"; then
    conda activate "$WANT" || true
    info "conda env: $WANT"
  fi
fi

# THE ACTIVATED ENVIRONMENT COMES FIRST.
#
# This list was written for inside the image, where /opt/conda is the only
# python that matters. Run natively with a conda env active, `command -v python`
# resolved to the BASE miniconda interpreter -- which has no torch -- and the
# search fell through to /usr/bin/python3 and gave up, while qb.sh had just
# reported "torch 2.8.0+cu128" from the very env it skipped. $CONDA_PREFIX and
# $VIRTUAL_ENV name the active environment directly and cannot be shadowed by
# PATH ordering, so they go at the front.
PY=""
for cand in ${PRE_PY:+"$PRE_PY"} \
            ${CONDA_PREFIX:+"$CONDA_PREFIX/bin/python"} \
            ${VIRTUAL_ENV:+"$VIRTUAL_ENV/bin/python"} \
            python python3 /opt/conda/envs/*/bin/python /opt/conda/bin/python \
            /usr/local/bin/python3 /usr/bin/python3; do
  c="$(command -v "$cand" 2>/dev/null || echo "$cand")"
  [ -x "$c" ] || continue
  if "$c" -c 'import torch, transformers' >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "FATAL: no interpreter found that imports both torch AND transformers."
  echo "tried:"
  echo "  (the environment active when you ran this: ${PRE_PY:-none found})"
  for cand in ${PRE_PY:+"$PRE_PY"} \
              ${CONDA_PREFIX:+"$CONDA_PREFIX/bin/python"} \
              ${VIRTUAL_ENV:+"$VIRTUAL_ENV/bin/python"} \
              python python3 /opt/conda/envs/*/bin/python /usr/bin/python3; do
    c="$(command -v "$cand" 2>/dev/null || echo "$cand")"
    if [ -x "$c" ]; then
      printf '   %-40s %s\n' "$c" \
        "$("$c" -c 'import torch,transformers;print("ok")' 2>&1 | tail -1)"
    fi
  done
  echo
  echo "  BOTH are required. A line above saying \"No module named 'transformers'\""
  echo "  means the environment is otherwise fine and one pip install fixes it:"
  echo "      pip install transformers"
  echo "  Activate the environment you want BEFORE running; \$CONDA_PREFIX is"
  echo "  checked first and is not affected by PATH ordering."
  exit 3
fi
info "python : $PY"
info "torch  : $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>&1 | tail -1)"
info "transf : $("$PY" -c 'import transformers;print(transformers.__version__)' 2>&1 | tail -1)"

if [ -z "$GPUS" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)"
fi
if [ "$GPUS" -lt 1 ]; then GPUS=1; fi

# Device for the preflight and the capability probe only. Derived from
# QB_GPU_LIST if the user named cards, otherwise 0. Kept in its OWN variable:
# defaulting QB_GPU_LIST itself made it permanently non-empty, which silently
# reduced every multi-GPU run to a single worker on device 0.
#
# The :- is load-bearing and its absence was a real crash. This script runs
# under `set -u`, and ${QB_GPU_LIST%%,*} on an UNSET variable is an unbound
# reference, not an empty string -- so `--gpus 1` (a count, which never sets
# QB_GPU_LIST) died with "QB_GPU_LIST: unbound variable" before loading
# anything. The old QB_GPU_LIST="${QB_GPU_LIST:-0}" line at the top of the file
# had been masking this, and removing it to fix the multi-GPU collapse exposed
# it. Substitute the default FIRST, then strip, then fall back to device 0.
PRE_DEV="${QB_GPU_LIST:-}"; PRE_DEV="${PRE_DEV%%,*}"; PRE_DEV="${PRE_DEV:-0}"

SUBSET="$QB/data/subset_full.jsonl"
[ -f "$SUBSET" ] || SUBSET="$QB/data/subset_200.jsonl"

# --------------------------------------------------------------------------- #
#  CHAT TEMPLATE -- resolve it ONCE, here, and hand it to everything downstream.
#
#  This was missing and the failure was absurd: the template is COPY'd into the
#  image at /opt/qb/data/chat_template.jinja by the build, but neither the
#  preflight nor the worker ever looked in that directory. They searched
#  $QB_CHAT_TEMPLATE, the model directory, and /work -- and this script never
#  set $QB_CHAT_TEMPLATE. So an image that contained the file refused to start
#  with "NO CHAT TEMPLATE anywhere", and the suggested fix (copy it into the
#  work dir) was busywork for a file already inside the image.
#
#  Worth 6.7 accuracy points, and its absence is silent at run time: the
#  tokenizer loads happily without it and the model then receives prompts in a
#  format it never saw in training. Hence resolve, export, and print.
# --------------------------------------------------------------------------- #
TPL=""
for c in "${QB_CHAT_TEMPLATE:-}" \
         "$QB/data/chat_template.jinja" \
         "/opt/qb/data/chat_template.jinja" \
         "$MODEL_DIR/chat_template.jinja" \
         "$WORK/chat_template.jinja"; do
  if [ -n "$c" ] && [ -f "$c" ]; then TPL="$c"; break; fi
done
if [ -n "$TPL" ]; then export QB_CHAT_TEMPLATE="$TPL"; fi

say "IMAGE B -- RetroDFM-R-8B, fp16 reference and quantized arms"
info "runner : $QB_RUNNER_REV"
info "model  : $MODEL_DIR"
info "template: ${TPL:-<none found; the tokenizer had better carry its own>}"
info "work   : $WORK"
info "GPUs   : $GPUS"
info "n      : $N_FULL   subset: $(basename "$SUBSET")"
date
if [ ! -d "$MODEL_DIR" ]; then
  warn "model directory not found: $MODEL_DIR -- searching for a checkpoint"
  # REQUIRE config.json AND weights AND a tokenizer. Accepting any directory
  # with a config.json is how a 16 KB stale HF cache entry -- two symlinks, no
  # weights, no tokenizer -- was once picked as the model, and three rounds
  # went into debugging a tokenizer that was never there.
  FOUND="$("$PY" - <<'PYFIND'
import glob, os
roots = [os.path.expanduser("~"), "/opt/model", "/scratch", "/models", os.getcwd()]
seen, best = set(), []
for r in roots:
    if not os.path.isdir(r):
        continue
    for cfg in glob.glob(os.path.join(r, "**", "config.json"), recursive=True):
        d = os.path.dirname(cfg)
        if d in seen or "/.git/" in d:
            continue
        seen.add(d)
        w = glob.glob(os.path.join(d, "*.safetensors")) or glob.glob(os.path.join(d, "*.bin"))
        t = (glob.glob(os.path.join(d, "tokenizer.json"))
             or glob.glob(os.path.join(d, "tokenizer_config.json")))
        if not (w and t):
            continue
        size = sum(os.path.getsize(x) for x in w if os.path.exists(x))
        name = os.path.basename(d).lower()
        # a RetroDFM-shaped name first, then the biggest set of weights
        best.append(((("retrodfm" in name or "retro" in name), size), d))
best.sort(reverse=True)
print(best[0][1] if best else "")
PYFIND
)"
  if [ -n "$FOUND" ]; then
    MODEL_DIR="$FOUND"
    warn "using discovered checkpoint: $MODEL_DIR"
    warn "  pass --model to override, or set QB_MODEL_DIR"
  else
    echo "no checkpoint found under \$HOME, /opt/model, /scratch, /models or \$PWD."
    echo "  A usable directory needs config.json, weights (*.safetensors or"
    echo "  *.bin) AND a tokenizer -- a config.json alone is not enough."
    echo "  Pass --model <dir>, or fetch it:"
    echo "    huggingface-cli download OpenDFM/RetroDFM-R-8B --local-dir ./retrodfm-r-8b"
    exit 2
  fi
fi
du -sh "$MODEL_DIR" | sed 's/^/   /' || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
  | awk 'NR<=3' || true

say "STAGE 0 -- preflight, then capability probe"
# Preflight: load the model on ONE gpu before launching sixteen of them. Every
# failure mode so far (no chat template, missing module, wrong model path,
# unreadable weights) shows up here in seconds instead of after a silent
# sixteen-way crash.
# `cmd <<'X' || { ... }` puts the || clause on the next line, which the heredoc
# then swallows as Python. Use if/then, where the redirect and the test stay on
# one logical line.
if ! CUDA_VISIBLE_DEVICES="$PRE_DEV" QB_MODEL_DIR="$MODEL_DIR" "$PY" -u - <<'PYPRE'
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
d = os.environ["QB_MODEL_DIR"]
tok = AutoTokenizer.from_pretrained(d)
src = "tokenizer" if getattr(tok, "chat_template", None) else ""
if not src:
    for c in [os.environ.get("QB_CHAT_TEMPLATE", ""),
              os.path.join(os.environ.get("QB_ROOT", "/opt/qb"), "data",
                           "chat_template.jinja"),
              "/opt/qb/data/chat_template.jinja",
              os.path.join(d, "chat_template.jinja"),
              "/work/chat_template.jinja"]:
        if c and os.path.isfile(c):
            src = c
            break
if not src:
    print("   ! NO CHAT TEMPLATE anywhere -- prompts would be malformed.", flush=True)
    print("     put chat_template.jinja in the work dir (bound at /work).", flush=True)
    sys.exit(1)
print(f"   chat template ok | from: {src}", flush=True)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
_cc = torch.cuda.get_device_capability(0)
_tag = f"sm_{_cc[0]}{_cc[1]}"
print(f"   gpu 0: {torch.cuda.get_device_name(0)}  {vram:.1f} GB  {_tag}", flush=True)

# THE IMAGE MUST BE ABLE TO TARGET THIS CARD AT ALL.
#
# Running an image built for another GPU generation produces a DIFFERENT,
# confusing failure for every level -- "PackageNotFoundError: bitsandbytes" here, "no
# kernel image is available" there -- and torch only whispers the real cause in
# a UserWarning that scrolls past. One check up front, and the message names
# the actual problem instead of leaving sixteen tracebacks to interpret.
_arch = list(torch.cuda.get_arch_list() or [])
if _arch and _tag not in _arch:
    print("", flush=True)
    print("   " + "=" * 68, flush=True)
    print(f"   FATAL: this image's PyTorch cannot target {_tag}.", flush=True)
    print("   " + "=" * 68, flush=True)
    print(f"     card         : {torch.cuda.get_device_name(0)}  ({_tag})", flush=True)
    print(f"     torch        : {torch.__version__}", flush=True)
    print(f"     built for    : {' '.join(_arch)}", flush=True)
    print("", flush=True)
    print("     Every level would fail, each with a different error, none of", flush=True)
    print("     which names this. Nothing below would be a measurement.", flush=True)
    print("", flush=True)
    print("     The image was probably built for another GPU generation.", flush=True)
    print("     Check which image is in this folder:", flush=True)
    print("        singularity exec <image>.sif cat /opt/qb/build-versions.json", flush=True)
    print("     and build one for this card with docker/build_images.sh.", flush=True)
    sys.exit(1)
if vram < 20:
    # The full-precision load below needs ~16 GB. On an 11 GB 2080 Ti or an
    # 8 GB 3070 that is guaranteed to OOM, and it would abort a run whose
    # quantized levels fit perfectly well -- bitsandbytes 4-bit never
    # materialises the base model at all. Check the config instead.
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(d)
    print(f"   config ok | {c.num_hidden_layers} layers, hidden {c.hidden_size}",
          flush=True)
    print("   small card: skipping the full-precision load test on purpose;",
          flush=True)
    print("   quantized levels stream in and do not need 16 GB.", flush=True)
else:
    # Deliberately NOT loading the weights. This check existed to catch a broken
    # model directory, but the image build already verifies the config, the
    # tokenizer and the shard index. What it added at run time was a 15.3 GB
    # load through device_map that aborts every worker whenever the card is
    # momentarily busy -- and, under some torch/accelerate combinations, leaves
    # parameters on the meta device and fails with "Cannot copy out of meta
    # tensor" instead of a clean OOM. It cost two evenings. The cheap checks
    # below catch the same real failures.
    from transformers import AutoConfig
    import glob, json
    c = AutoConfig.from_pretrained(d)
    print(f"   config ok | {c.num_hidden_layers} layers, hidden {c.hidden_size}",
          flush=True)
    shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    gb = sum(os.path.getsize(p) for p in shards) / 1024**3
    print(f"   weights   | {len(shards)} shards, {gb:.1f} GB", flush=True)
    if not shards or gb < 10:
        print("   ! model directory looks truncated", flush=True); sys.exit(1)
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        want = set(json.load(open(idx))["weight_map"].values())
        have = {os.path.basename(p) for p in shards}
        if want - have:
            print(f"   ! index references missing shards: {sorted(want-have)}", flush=True)
            sys.exit(1)
        print("   shard index complete", flush=True)
    free, total = torch.cuda.mem_get_info(0)
    print(f"   gpu free  | {free/1024**3:.1f} of {total/1024**3:.1f} GB", flush=True)
    if free / 1024**3 < 17:
        print("   ! less than 17 GB free -- another process is using this card.",
              flush=True)
        print("     16-bit levels will not fit. Free it or pick another GPU.",
              flush=True)
PYPRE
then
  warn "PREFLIGHT FAILED -- not launching workers"
  exit 4
fi

CUDA_VISIBLE_DEVICES="$PRE_DEV" "$PY" -u "$QB/scripts/probe_capability.py" \
    "$WORK/probe/capability.json" || warn "probe failed"

if [ -z "$LEVELS" ]; then
  LEVELS="fp16"
  if [ -f "$WORK/probe/capability.json" ]; then
    "$PY" -c "
import json;d=json.load(open('$WORK/probe/capability.json'))
print(1 if d.get('int8_weight_only',{}).get('ok') else 0)" 2>/dev/null | grep -q 1 \
      && LEVELS="$LEVELS int8_ao"
    "$PY" -c "
import json;d=json.load(open('$WORK/probe/capability.json'))
print(1 if d.get('w8a8',{}).get('ok') else 0)" 2>/dev/null | grep -q 1 \
      && LEVELS="$LEVELS w8a8"
  fi
fi
info "levels : $LEVELS"

# Ask the interpreter what this build can actually do, once, before launching
# workers. bitsandbytes missing is the common case: in an image built without
# it, every nf4/nf4dq/fp4 worker dies with
# "PackageNotFoundError: bitsandbytes" inside transformers -- N identical
# tracebacks for one missing package. Drop the level, say why, keep going.
KEEP=""; DROPPED=""
for LV in $LEVELS; do
  case "$LV" in
    nf4|nf4dq|fp4)
      if "$PY" -c 'import bitsandbytes' >/dev/null 2>&1; then KEEP="$KEEP $LV"
      else DROPPED="$DROPPED $LV(bitsandbytes not in this image)"; fi ;;
    *) KEEP="$KEEP $LV" ;;
  esac
done
if [ -n "${DROPPED// /}" ]; then
  warn "these levels cannot run in this image and were dropped BEFORE launching:"
  for d in $DROPPED; do warn "    $d"; done
  warn "  an image that includes bitsandbytes:"
  warn "    docker/build_images.sh --mslk"
fi
LEVELS="${KEEP# }"
if [ -z "${LEVELS// /}" ]; then
  warn "no runnable levels left -- nothing to do"
  exit 7
fi

if [ "$DRY" = 1 ]; then say "dry run -- stopping"; exit 0; fi

# Augmented runs must not write into the un-augmented run's files. The worker
# resumes by uid, so a k=5x1x2 run started on top of a finished 1x1x1 run would
# see every uid already present, skip all 5005 molecules and exit "successful"
# with the old single-candidate results still on disk. The k's therefore go in
# the filename: fp16.s0.jsonl vs fp16.k5x1x2.s0.jsonl.
KA="${QB_KA:-1}"; KS="${QB_KS:-1}"; KB="${QB_KB:-1}"
export QB_MEM_LIMIT_GB="${QB_MEM_LIMIT_GB:-0}"
export QB_MEM_LIMIT_PHASE="${QB_MEM_LIMIT_PHASE:-all}"
export QB_OFFLOAD="${QB_OFFLOAD:-0}"
export QB_AUG="${QB_AUG:-canonical}"
export QB_GEN_CHUNK="${QB_GEN_CHUNK:-0}"
export QB_ANS_CHUNK="${QB_ANS_CHUNK:-0}"
SEED="${QB_SEED:-1234}"; GREEDY="${QB_GREEDY:-0}"   # --greedy sets QB_GREEDY
# 0 = top-k filtering off, matching the reference implementation. The
# checkpoint's generation_config sets 20; QB_TOP_K=20 uses that instead.
# Never -1: in transformers that means greedy, not disabled.
TOPK="${QB_TOP_K:-0}"
KTAG=""
if [ $((KA * KS * KB)) -gt 1 ]; then KTAG=".k${KA}x${KS}x${KB}"; fi
# Decoding mode and seed go in the filename for the same reason the k's do: a
# control run must not land on top of the run it is the control for.
if [ "$GREEDY" = 1 ];    then KTAG="$KTAG.greedy";      fi
if [ "$SEED" != 1234 ];  then KTAG="$KTAG.seed$SEED";   fi
# top_k likewise. The record fingerprint already distinguishes 0 from 20, so
# the MERGED file would be correct either way -- but without this the two runs
# share a shard filename and the merge has to discard half of what it reads.
if [ "$TOPK" != 0 ];     then KTAG="$KTAG.topk$TOPK";   fi
# An offloaded run streams layers over PCIe, so its timings mean something
# different. Same reasoning as greedy and seed: keep it in its own files.
if [ "$QB_OFFLOAD" = 1 ]; then KTAG="$KTAG.offload";      fi
# A chunked run is not bit-identical to an unchunked one, so it gets its own
# filename for the same reason greedy and seed do.
if [ "$QB_GEN_CHUNK" != 0 ]; then KTAG="$KTAG.gc$QB_GEN_CHUNK"; fi
if [ "$QB_ANS_CHUNK" != 0 ]; then KTAG="$KTAG.ac$QB_ANS_CHUNK"; fi
# The augmentation mode belongs here for exactly the reason greedy/seed/topk/gc
# do: it changes the RECORD fingerprint, and a record fingerprint alone is not
# enough. Two runs differing only in QB_AUG used to land on the SAME shard
# filenames; load_done() then saw records from "a DIFFERENT configuration",
# renamed the whole file to .stale-<timestamp> and regenerated from scratch.
# Running a permute arm beside an existing canonical one would therefore have
# moved days of finished work aside, mid-run, with no warning. Distinct tag,
# distinct files, no collision.
if [ "${QB_AUG:-canonical}" != canonical ]; then KTAG="$KTAG.aug$QB_AUG"; fi
# The k_b OPERATOR belongs here too, and for a sharper reason than the rest.
# k_b changed between paper versions: the old one is a partial beam search from
# <answer> at T=0 ranked by Eq.5 with alpha=1, the new one samples at T=1.4 and
# counts plain frequency (Eq.6). Every arm run before 2026-09-06 used the old
# one; the default is now the new one. Those are different METHODS, not
# settings -- but they share a level and a budget, so without this a beam
# control run and the sampling arm it is the control FOR land on the same shard
# filenames, and load_done() renames whichever it finds to .stale-<timestamp>.
# Days of finished work moved aside mid-run, with a rename to show for it.
if [ "${QB_ANS_MODE:-sample}" != sample ]; then KTAG="$KTAG.ans${QB_ANS_MODE}"; fi
# alpha only means anything to the beam operator; the sampling operator ignores
# it. Tag it when it is not the paper's 0.0 so a re-ranked beam arm and a plain
# one are distinguishable on disk.
if [ "${QB_ALPHA:-0.0}" != "0.0" ]; then KTAG="$KTAG.a${QB_ALPHA}"; fi
case "${QB_BASE_DTYPE:-bf16}" in
  bf16) ;;
  fp16) KTAG="$KTAG.basefp16" ;;
  *) echo "--base must be bf16 or fp16, got '${QB_BASE_DTYPE}'" >&2; exit 2 ;;
esac

# Warn BEFORE the run, not after six hours of thrashing. Peak KV cache during
# the k_s stage is roughly k_s x max_new_tokens x 150 kB for this 8B model;
# with k_s=10 and 2048 tokens that is ~3 GB on top of the weights. On an 11 GB
# card it fits only just, and the allocator spends the run recovering from
# near-misses -- measured at 2500 s/mol against 235 s/mol on a big card.
if [ "$QB_GEN_CHUNK" = 0 ] && [ "$KS" -gt 2 ]; then
  VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 || echo 0)"
  VRAM_GB=$((VRAM_GB / 1024))
  # ~150 kB per token per sequence for this 8B model at 2048 tokens => ~0.3 GB
  # per sampled path. Integer arithmetic only; bash has no floats.
  KVGB=$(( KS * 3 / 10 ))
  if [ "$VRAM_GB" -gt 0 ] && [ "$VRAM_GB" -lt 16 ]; then
    warn "smallest card is ${VRAM_GB} GB and k_s=$KS asks for all $KS paths in one call"
    warn "  (~${KVGB} GB of KV cache on top of the weights)"
    warn "  if this thrashes, re-run with:  --gen-chunk 2"
    warn "  same candidates, ~$((KS / 2))x less peak memory, different random order"
  fi
fi
info "augmentation: $QB_AUG (root atom + traversal order)"
# Print the k_b operator every run. It is the difference between the current
# paper's method and the previous one, it is invisible in the level name, and
# an arm quoted next to the wrong ladder is a wasted run nobody notices.
if [ "${QB_ANS_MODE:-sample}" = sample ]; then
  info "k_b operator: NEW -- ${QB_KB:-$KB} samples at T=${QB_ANS_TEMP:-1.4}, Eq.6 frequency, alpha=${QB_ALPHA:-0.0}"
else
  info "k_b operator: OLD -- partial beam from <answer> at T=0, Eq.5 alpha=${QB_ALPHA:-0.0}  (pre-2026-09-06 arms)"
fi
info "compute  : ${QB_BASE_DTYPE:-bf16}$( [ "${QB_BASE_DTYPE:-bf16}" = fp16 ] && echo '  (tagged .basefp16 on disk)')"
# Greedy draws the same answer every time, so asking for k_b of them buys
# nothing but wall clock. Say so before the run rather than after.
if [ "$GREEDY" = 1 ] && [ "$KB" -gt 1 ]; then
  warn "--greedy with --kb $KB: every sample would be identical, so only one is taken. Use --kb 1."
fi
info "k_a x k_s x k_b : ${KA} x ${KS} x ${KB}$([ -n "$KTAG" ] || echo '   (un-augmented baseline, paper Table 1a = 59.0%)')"
info "decoding : $([ "$GREEDY" = 1 ] && echo 'greedy' || echo "sampled, seed $SEED, temp ${QB_TEMP_S:-1.0}, top_p ${QB_TOP_P:-1.0}, top_k $TOPK$([ "$TOPK" = 0 ] && echo ' (off)')")"

DONE_LEVELS=""; FAILED_LEVELS=""
for LV in $LEVELS; do
  TAG="$LV$KTAG"

  # --- --purge-failed: delete this level's failures, then stop ---------------
  # TAG is already built from --levels/--ka/--ks/--kb above, including .greedy,
  # .seed, .topk, .offload and .gc, so the right files are selected by the same
  # flags that produced them. Nothing is re-derived here and nothing can drift.
  if [ "$PURGE" = 1 ]; then
    python3 - "$WORK" "$TAG" <<'PY'
import glob, json, os, shutil, sys

work, tag = sys.argv[1], sys.argv[2]

# NEVER purge underneath a live worker. The purge reads a shard file, drops the
# error lines and atomically replaces it, so any record a worker appends between
# the read and the replace is lost -- a run of days would come back short with
# nothing to show why. The plan is "let the current job finish, THEN clean up";
# this makes doing it a few minutes early impossible rather than inadvisable.
#
# Checked through /proc rather than `pgrep -f run_llm.py`, which matches ANY
# command line containing that text: the shell running this purge, a grep, an
# editor with the file open, a `tail -f`. That guard cried wolf on its first
# test. Here the process must actually BE a python interpreter running the
# worker, and our own process tree is excluded.
def live_workers():
    me = {os.getpid(), os.getppid()}
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in me:
            continue
        try:
            with open("/proc/%s/comm" % d) as f:
                comm = f.read().strip()
            if not comm.startswith("python"):
                continue
            with open("/proc/%s/cmdline" % d, "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue                      # it exited while we looked: not live
        if any(a.endswith(b"run_llm.py") for a in argv if a):
            out.append((d, b" ".join(a for a in argv if a).decode("utf-8", "replace")[:120]))
    return out


busy = live_workers()
if busy and os.environ.get("QB_PURGE_FORCE", "0") != "1":
    print("=" * 74)
    print("REFUSING TO PURGE -- a worker is still running")
    print("=" * 74)
    for pid, cl in busy:
        print("  pid %-8s %s" % (pid, cl))
    print()
    print("  Purging now would drop records written while the files are rewritten.")
    print("  Wait for the run to finish, then repeat the same command.")
    print("  Watch it with:  tail -f %s" % os.path.join(work, "logs", tag + ".s0.log"))
    print()
    print("  Override only if those processes are certainly unrelated:")
    print("      QB_PURGE_FORCE=1 <same command>")
    raise SystemExit(3)
if busy:
    print("  QB_PURGE_FORCE=1 set -- purging with %d worker(s) alive." % len(busy))

files  = sorted(glob.glob(os.path.join(work, "shards", tag + ".s*.jsonl")))
files += sorted(glob.glob(os.path.join(work, "raw", "retrodfm-r-8b.%s.retro.jsonl" % tag)))

print("=" * 74)
print("PURGE FAILED MOLECULES  --  level tag [%s]" % tag)
print("=" * 74)
if not files:
    print("  no files for this level in %s" % work)
    print("  (check --levels/--ka/--ks/--kb match the run you want to clean)")
    raise SystemExit(0)

gone = {"shards": 0, "raw": 0}
kept = {"shards": 0, "raw": 0}
touched = 0
for q in files:
    where = "raw" if os.sep + "raw" + os.sep in q else "shards"
    keep, removed, bad = [], 0, 0
    for line in open(q):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            bad += 1                       # half-written last line after a kill
            continue
        if str(rec.get("stop_reason", "")).startswith("error"):
            removed += 1
        else:
            keep.append(line if line.endswith("\n") else line + "\n")
    if removed == 0 and bad == 0:
        print("  %-52s %6d kept, nothing to purge" % (os.path.basename(q), len(keep)))
        kept[where] += len(keep)
        continue
    bak = q + ".before-purge"
    if not os.path.exists(bak):            # never overwrite the first backup
        shutil.copy2(q, bak)
    with open(q + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(q + ".tmp", q)              # atomic: never a half-written result file
    extra = ", %d unparsable line(s) dropped" % bad if bad else ""
    print("  %-52s %6d kept, %5d DELETED%s"
          % (os.path.basename(q), len(keep), removed, extra))
    gone[where] += removed
    kept[where] += len(keep)
    touched += 1

print("-" * 74)
# shards/ and raw/ hold the SAME molecules. Reported separately, because adding
# them together doubles the number and reads as though twice as much was lost.
print("  shards/ : %6d failed deleted, %6d answers kept" % (gone["shards"], kept["shards"]))
print("  raw/    : %6d failed deleted, %6d answers kept" % (gone["raw"], kept["raw"]))
print("  %d file(s) rewritten%s"
      % (touched, "; originals beside them as *.before-purge" if touched else ""))
print()
print("  Those molecules have no record now, so the next run recomputes exactly")
print("  them. Re-run the same command without --purge-failed.")
PY
    continue
  fi

  # QB_GPU_LIST="2,5,9" selects physical cards. $g remains the SHARD
  # index, so filenames, QB_SHARD and resume logic are unchanged.
  IFS="," read -ra QB_DEVS <<< "${QB_GPU_LIST:-$(seq -s, 0 $((GPUS - 1)))}"
  GPUS="${#QB_DEVS[@]}"
  say "LEVEL $LV -- $N_FULL molecules over $GPUS GPUs   [$TAG]  devices: ${QB_DEVS[*]}"
  pids=()
  for g in $(seq 0 $((GPUS - 1))); do
    (
      # APPEND, never truncate. The worker prints "FIRST ERROR -- full
      # traceback" the first time a molecule raises, and that traceback is
      # the only record of WHICH operation faulted. This redirect used to be
      # `>`, so the next restart wiped it: on 2026-09-10 three shards faulted
      # with a CUDA error, and the two restart attempts that followed
      # truncated all three logs to 53 bytes before anyone read them. The
      # evidence existed and the retry destroyed it. Keep every session.
      {
        echo
        echo "############################################################"
        echo "# session $(date '+%Y-%m-%d %H:%M:%S %Z')  shard $g/$GPUS  level $LV  tag $TAG"
        echo "# host $(hostname)  device ${QB_DEVS[$g]}"
        echo "############################################################"
      } >> "$WORK/logs/$TAG.s$g.log" 2>&1
      CUDA_VISIBLE_DEVICES="${QB_DEVS[$g]}" \
      QB_MODEL_DIR="$MODEL_DIR" QB_LEVEL="$LV" QB_N="$N_FULL" \
      QB_SUBSET="$SUBSET" QB_SHARD="$g/$GPUS" \
      QB_OUT="$WORK/shards/$TAG.s$g.jsonl" \
      QB_KA="$KA" QB_KS="$KS" QB_KB="$KB" \
      QB_SEED="$SEED" QB_GREEDY="$GREEDY" QB_TOP_K="$TOPK" \
      QB_MEM_LIMIT_GB="$QB_MEM_LIMIT_GB" \
      QB_MEM_LIMIT_PHASE="$QB_MEM_LIMIT_PHASE" QB_OFFLOAD="$QB_OFFLOAD" \
      QB_AUG="$QB_AUG" QB_GEN_CHUNK="$QB_GEN_CHUNK" \
      QB_ANS_CHUNK="$QB_ANS_CHUNK" \
      QB_ANS_MODE="${QB_ANS_MODE:-sample}" QB_ALPHA="${QB_ALPHA:-0.0}" \
      QB_ANS_TEMP="${QB_ANS_TEMP:-1.4}" QB_THINK_TEMP="${QB_THINK_TEMP:-1.1}" \
      QB_BASE_DTYPE="${QB_BASE_DTYPE:-bf16}" \
      "$PY" -u "$QB/scripts/run_llm.py" \
        >> "$WORK/logs/$TAG.s$g.log" 2>&1
    ) &
    pids+=($!)
  done
  info "launched ${#pids[@]} workers; tail $WORK/logs/$TAG.s0.log to watch"
  # Name the shard and the exit code. "a worker exited non-zero" said neither
  # WHICH worker nor HOW it failed, so two identical lines under a "2 GPUs"
  # header read as though the worker count were the problem. It never was --
  # the code says whether the process ran and failed (1, 8) or was killed
  # (>128), and the shard says which log holds the traceback.
  si=0
  for p in "${pids[@]}"; do
    rc=0; wait "$p" || rc=$?
    if [ "$rc" -ne 0 ]; then
      if [ "$rc" -gt 128 ]; then
        SIG=$((rc - 128))
        warn "shard s$si was KILLED by signal $SIG -- see $WORK/logs/$TAG.s$si.log"
        case "$SIG" in
          9)  warn "  signal 9 is almost always the out-of-memory killer (host RAM, not VRAM)" ;;
          # SIGBUS is NOT out of memory. The weights are memory-mapped, so this
          # is a failed page fault on the file: the .sif truncated or still
          # copying, the disk full, or /dev/shm too small. It fires while
          # loading, before any GPU work, so it looks identical for every level
          # and is unrelated to whatever level was requested.
          7)  warn "  signal 7 is SIGBUS -- a memory-MAPPING failure, not a memory shortage."
              warn "  The weights are memory-mapped, so it is the FILE that could not"
              warn "  be read -- nothing to do with GPU memory."
              warn "  Check, in this order:"
              warn "    1. the .sif is complete:  sha256sum on BOTH machines must match"
              warn "       (an image still being copied fails exactly here)"
              warn "    2. free disk:             df -h / /tmp \"$WORK\""
              warn "    3. shared memory:         df -h /dev/shm   (needs a few GB)"
              warn "    4. storage errors:        dmesg | tail -30" ;;
          15) warn "  signal 15 is SIGTERM -- something asked it to stop (scheduler? Ctrl-C?)" ;;
        esac
      else
        warn "shard s$si RAN AND FAILED, exit $rc -- traceback in $WORK/logs/$TAG.s$si.log"
      fi
    fi
    si=$((si + 1))
  done

  # Merge shards: de-duplicate by uid, and keep ONLY records carrying this
  # run's fingerprint. Shard files from earlier runs of the same level sit in
  # the same directory -- a 2-GPU rerun once merged 4383 records from a
  # previous 16-GPU run and reported them as the result.
  # `|| MERGE_RC=$?` matters: with `set -e`, a bare failing command exits the
  # script immediately, so the guard below would never run and the useful
  # diagnostic -- which log to read -- would never be printed. Capturing the
  # status inside a condition suppresses that.
  MERGE_RC=0
  "$PY" - "$WORK/shards" "$TAG" "$WORK/raw" <<'PY' || MERGE_RC=$?
import glob, json, os, sys
from collections import Counter
shards, lv, dst = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)

cfgs = sorted(glob.glob(os.path.join(shards, f"{lv}.s*.jsonl.cfg")))
want = open(cfgs[0]).read().strip() if cfgs else None
if want is None:
    print("   ! no .cfg sidecar; merging everything (pre-fingerprint data)")

# A MOLECULE THAT FAILED AND WAS THEN REDONE MUST COME BACK AS THE GOOD ONE.
#
# This used to keep whichever record for a uid appeared FIRST. Since a retry is
# appended after the failure it replaces, first-wins threw the retry away and
# kept the error -- so re-running recovered nothing, however many times it ran.
#
# Rank: a real answer beats an error; between two of equal rank the later one
# wins, because it is the more recent attempt. A molecule that only ever failed
# still keeps its error record, so scoring can count it in `excluded` instead of
# silently shrinking the denominator.
best, tally = {}, Counter()
for p in sorted(glob.glob(os.path.join(shards, f"{lv}.s*.jsonl"))):
    for line in open(p):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        cfg = rec.get("cfg", "<none>")
        tally[cfg] += 1
        if want is not None and cfg != want:
            continue
        rank = 0 if str(rec.get("stop_reason", "")).startswith("error") else 1
        prev = best.get(rec["uid"])
        if prev is None or rank >= prev[0]:
            best[rec["uid"]] = (rank, line)

out_path = os.path.join(dst, f"retrodfm-r-8b.{lv}.retro.jsonl")
with open(out_path, "w") as out:
    for rank, line in best.values():
        out.write(line)
kept = len(best)
bad = sum(1 for rank, _ in best.values() if rank == 0)

if len(tally) > 1:
    print("   shard directory holds more than one configuration:")
    for c, k in tally.most_common():
        print(f"      {k:6d}  {'KEPT  ' if c == want else 'skipped'}  {c}")
print(f"   merged {kept} records -> retrodfm-r-8b.{lv}.retro.jsonl")
if bad:
    print(f"   {bad} of them are still errors -- rerun the same command to "
          f"retry just those; see logs/{lv}.s*.log for the reason")
if kept == 0:
    sys.exit(7)
PY
  # A worker that dies on import writes nothing, and an unguarded run would
  # sail on to scoring and report an empty result as if it were data.
  #
  # RECORD the failure and CONTINUE, rather than exiting. A sweep across a
  # dozen levels always contains some the build or the card cannot do -- w4a8
  # and w4a4 are absent from torchao 0.18, nvfp4 needs sm_100 -- and aborting
  # on the first of them throws away every level queued behind it. That is
  # hours of GPU time discarded because of a level that was never going to
  # work. The guard that matters is the one against reporting an empty result
  # AS data, and that is preserved: the level is excluded and named at the end,
  # and the script still exits non-zero so a wrapper can tell.
  if [ "$MERGE_RC" -ne 0 ] || [ ! -s "$WORK/raw/retrodfm-r-8b.$TAG.retro.jsonl" ]; then
    warn "level $TAG produced NO records for this configuration -- skipping it."
    warn "last lines of its first worker log:"
    tail -n 12 "$WORK/logs/$TAG.s0.log" 2>/dev/null | sed 's/^/      /'
    FAILED_LEVELS="$FAILED_LEVELS $TAG"
    rm -f "$WORK/raw/retrodfm-r-8b.$TAG.retro.jsonl"   # never score an empty file
    continue
  fi
  DONE_LEVELS="$DONE_LEVELS $TAG"
done

if [ -n "${DONE_LEVELS// /}" ]; then
  info "levels with results:$DONE_LEVELS"
fi
if [ -n "${FAILED_LEVELS// /}" ]; then
  warn "levels with NO results:$FAILED_LEVELS"
  warn "read $WORK/probe/capability.json to see whether the cause was the build"
  warn "or the architecture -- the two have different remedies."
fi

say "SCORING"
SC=0; "$PY" -u "$QB/scripts/score_sweep.py" "$WORK/raw" || SC=$?
if [ "$SC" -ne 0 ] && [ "$SC" -ne 3 ]; then warn "scoring failed (exit $SC)"; fi

ZIP="$WORK/retro-quantbench-llm.$(date +%Y%m%d).zip"
( cd "$WORK" && zip -qr "$ZIP" raw probe logs ./*.json 2>/dev/null ) || warn "zip failed"
if [ -f "$ZIP" ]; then info "package: $ZIP ($(du -h "$ZIP" | cut -f1))"; fi

say "COMPLETE"
# Do not congratulate the run on a reference it does not have. When every level
# failed, this epilogue printed underneath two EMPTY tables and read as though
# the sweep had succeeded -- the strongest wrong signal in the whole log.
if [ -z "${DONE_LEVELS// /}" ]; then
  warn "NOTHING RAN. No level produced a single record, so the tables above are"
  warn "empty -- they are not a result of zero accuracy, they are no data."
  warn "Start from the first error in $WORK/logs/, not from this line."
else
  # This used to announce "the fp16 rows are the true full-precision
  # reference..." on EVERY successful run, including runs with no fp16 level in
  # them at all -- a claim about data that was never produced, printed directly
  # under a table that did not contain it. Say what this run actually did.
  info "levels with results: $DONE_LEVELS"
  case " $DONE_LEVELS " in
    *" fp16"*|*" bf16"*)
      info "a full-precision arm ran, so the quantized rows in this sweep have"
      info "their own reference rather than borrowing one from another run." ;;
    *)
      info "NO full-precision arm in this sweep, so the 'changed' and McNemar"
      info "columns are empty by construction -- they need bf16 (or fp16) at"
      info "the SAME budget to compare against. Add one, or score against an"
      info "earlier sweep with --results." ;;
  esac
  if [ $((KA * KS * KB)) -eq 1 ]; then
    info "1x1x1 produces exactly ONE candidate per molecule, so top-3, top-5"
    info "and top-10 in the table above are equal to top-1 by construction."
    info "Only the top-1 column means anything for this arm; it is the number"
    info "to set beside the paper's un-augmented Table 1a (59.0%)."
  fi
fi
date

# Non-zero if any level produced nothing, so an unattended wrapper can tell the
# difference between "finished" and "finished, but three levels are missing".
if [ -n "${FAILED_LEVELS// /}" ]; then exit 6; fi
