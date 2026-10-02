#!/usr/bin/env bash
# ===========================================================================
#  Build the RetroDFM-R-8B image and convert it to Singularity.
#
#      bash docker/build_images.sh --mslk --model-dir <weights>
#
#  One image runs both inference settings:
#
#      torch 2.13.0+cu132, torchao 0.18.0, bitsandbytes 0.50.2, MSLK 1.3.0,
#      Python 3.12 and transformers 4.57.6 on a CUDA 13.3 base
#      -> ~/retro-llm-mslk.sif
#
#  --mslk selects exactly these versions. They are also the defaults, so the
#  flag can be left out. bitsandbytes provides nf4, nf4dq and fp4, torchao
#  provides int8_ao, and MSLK is needed only by torchao formats that are not
#  part of the paper. The image is about 25 GB.
#
#  It goes to your Linux home, never to /mnt/c: the 9p mount is 10-50x slower
#  and a 5 GB context transfer across it aborts with "context canceled".
#
#  The build fails, rather than a later run, if torch, transformers or torchao
#  do not import, the model config or tokenizer cannot be read, the shard
#  index is incomplete, or the weights are smaller than 10 GB.
#
#  REQUIREMENTS
#      docker            (Docker Desktop, WSL integration on)
#      apptainer or singularity
#      ~60 GB free for intermediate layers
#
#  NOTHING IS INSTALLED ON THE MACHINE THAT RUNS THE IMAGE
#  ------------------------------------------------------
#  Python, torch, torchao, bitsandbytes, transformers, rdkit, the model weights
#  and the test set all go INSIDE the .sif during this build. The run machine
#  needs a GPU driver and singularity and nothing else. The resolved versions
#  are written into the image at /opt/qb/build-versions.json and
#  /opt/qb/pip-freeze.txt.
#
#  Options
#      --skip-sif      build the docker image only
#      --model-dir D   RetroDFM-R weights (default ~/retro/retrodfmr-hf)
#      --cuda TAG      CUDA base image tag
#      --torch V       version or "latest" (same for --torchao, --bnb)
#      --bnb none      skip bitsandbytes entirely
# ===========================================================================
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CUDA_TAG="13.3.0-cudnn"
CUDA_DISTRO="devel-ubuntu24.04"
PY_VER="3.12"
TORCH_CUDA="cu132"
TORCH_VER="2.13.0"
TORCHAO_VER="0.18.0"
TORCHAO_INDEX="cuda"
BNB_VER="0.50.2"
MSLK_VER="1.3.0"
LLM_TAG="retro-llm:mslk"
LLM_OUT="$HOME/retro-llm-mslk.sif"
MODEL_SRC="$HOME/retro/retrodfmr-hf"
SKIP_SIF=0

while [ $# -gt 0 ]; do
  case "$1" in
    # The preset is the whole image in one flag instead of six that have to
    # agree with each other: a build that gets any one of them wrong fails
    # only once it reaches a GPU -- which is on the other machine, hours later.
    # CUDA 13.3 base with cu132 wheels: there is no cu133 wheel index, and
    # MSLK 1.3.0 supports CUDA 13.0 and 13.2 only -- so cu132 is the newest
    # matched stack. The base image version is nearly irrelevant here because
    # the wheels carry their own CUDA runtime; what must match is the wheel
    # index and MSLK.
    --mslk)
      CUDA_TAG="13.3.0-cudnn"; CUDA_DISTRO="devel-ubuntu24.04"; PY_VER="3.12"
      TORCH_CUDA="cu132"
      TORCH_VER="2.13.0"; TORCHAO_VER="0.18.0"; TORCHAO_INDEX="cuda"
      BNB_VER="0.50.2"; MSLK_VER="1.3.0"
      LLM_TAG="retro-llm:mslk"; LLM_OUT="$HOME/retro-llm-mslk.sif"
      shift ;;
    --skip-sif)  SKIP_SIF=1; shift ;;
    --model-dir) MODEL_SRC="$2"; shift 2 ;;
    --cuda)       CUDA_TAG="$2";   shift 2 ;;
    --cuda-distro) CUDA_DISTRO="$2"; shift 2 ;;
    --python)     PY_VER="$2";     shift 2 ;;
    --torch-cuda) TORCH_CUDA="$2"; shift 2 ;;
    --torch)      TORCH_VER="$2";  shift 2 ;;
    --torchao)    TORCHAO_VER="$2"; shift 2 ;;
    --torchao-index) TORCHAO_INDEX="$2"; shift 2 ;;   # pypi | cuda
    --bnb)        BNB_VER="$2";     shift 2 ;;
    --mslk-ver)   MSLK_VER="$2";    shift 2 ;;
    --tag)        LLM_TAG="$2";    shift 2 ;;
    --out)        LLM_OUT="$2";    shift 2 ;;
    -h|--help)   sed -n '2,43p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m   ! %s\033[0m\n' "$*"; }
die()  { printf '\033[31m   FAILED: %s\033[0m\n' "$*"; exit 1; }

command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "docker daemon unreachable -- start Docker Desktop"
SING="$(command -v singularity || command -v apptainer || true)"
[ "$SKIP_SIF" = 0 ] && [ -z "$SING" ] && \
  die "no singularity/apptainer. Install: sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt install -y apptainer"

STAGE="$HOME/.cache/qb-build"          # ext4, never /mnt/c
rm -rf "$STAGE"; mkdir -p "$STAGE"
say "staging on ext4: $STAGE"

# --------------------------------------------------------------------------- #
convert () {                            # $1 tag  $2 output .sif
  local tag="$1" out="$2"
  [ "$SKIP_SIF" = 1 ] && { info "--skip-sif: docker image '$tag' is ready"; return 0; }
  local tar="$STAGE/$(basename "$out" .sif).tar"
  info "exporting $tag ..."
  docker save "$tag" -o "$tar"
  info "archive $(du -h "$tar" | cut -f1); converting with $(basename "$SING") ..."
  "$SING" build --force "$out" "docker-archive://$tar" || die "sif build failed for $tag"
  rm -f "$tar"
  info "wrote $out ($(du -h "$out" | cut -f1))"
}

# --------------------------------------------------------------------------- #
build_llm () {
  say "RetroDFM-R-8B image"
  [ -d "$MODEL_SRC" ] || die "weights not found: $MODEL_SRC (use --model-dir)"
  ls "$MODEL_SRC"/*.safetensors >/dev/null 2>&1 || \
    die "no *.safetensors in $MODEL_SRC"

  local ctx="$STAGE/llm"; mkdir -p "$ctx/model"
  info "copying weights ($(du -sh "$MODEL_SRC" | cut -f1)) -- several minutes"
  cp "$MODEL_SRC"/*.safetensors "$MODEL_SRC"/*.json "$ctx/model/" 2>/dev/null || true
  for f in tokenizer.model tokenizer.json vocab.json merges.txt; do
    [ -f "$MODEL_SRC/$f" ] && cp "$MODEL_SRC/$f" "$ctx/model/"
  done
  info "model dir $(du -sh "$ctx/model" | cut -f1), $(ls "$ctx/model"/*.safetensors | wc -l) shards"

  # The image holds the same files as run/: Python under /opt/qb/scripts,
  # shell scripts under /opt/qb/dist, the test set and the chat template under
  # /opt/qb/data. qb.sh binds the copies in run/ over these at run time.
  mkdir -p "$ctx/data" "$ctx/scripts" "$ctx/dist"
  cp "$ROOT/data/subset_full.jsonl" "$ROOT/run/chat_template.jinja" "$ctx/data/"
  cp "$ROOT"/run/*.sh "$ctx/dist/"
  for f in "$ROOT"/run/*.py; do
    case "$(basename "$f")" in
      qb-run_llm.py) cp "$f" "$ctx/scripts/run_llm.py" ;;
      *)             cp "$f" "$ctx/scripts/" ;;
    esac
  done
  cp "$HERE/Dockerfile.llm" "$ctx/Dockerfile"

  info "base    : nvidia/cuda:$CUDA_TAG-$CUDA_DISTRO"
  info "python  : $PY_VER"
  info "torch   : $TORCH_VER  from https://download.pytorch.org/whl/$TORCH_CUDA"
  info "torchao : $TORCHAO_VER from $TORCHAO_INDEX    bitsandbytes: ${BNB_VER:-skipped}   mslk: ${MSLK_VER:-skipped}"
  info "everything above is installed INTO the image; the run machine needs none of it"
  docker build --build-arg "CUDA_TAG=$CUDA_TAG" \
      --build-arg "CUDA_DISTRO=$CUDA_DISTRO" \
      --build-arg "PY_VER=$PY_VER" \
      --build-arg "TORCH_CUDA=$TORCH_CUDA" \
      --build-arg "TORCH_VER=$TORCH_VER" \
      --build-arg "TORCHAO_VER=$TORCHAO_VER" \
      --build-arg "TORCHAO_INDEX=$TORCHAO_INDEX" \
      --build-arg "BNB_VER=$BNB_VER" \
      --build-arg "MSLK_VER=$MSLK_VER" \
      -t "$LLM_TAG" "$ctx" || die "image B build failed"
  convert "$LLM_TAG" "$LLM_OUT"

  # Print what the build actually resolved. With --torch latest this is the only
  # place the number appears, and it is the number the paper has to quote.
  if [ "$SKIP_SIF" = 0 ]; then
    info "versions baked into $LLM_OUT:"
    "$SING" exec "$LLM_OUT" cat /opt/qb/build-versions.json 2>/dev/null | sed 's/^/     /' || true

    # ------------------------------------------------------------------ #
    #  READ THE WEIGHTS BACK BEFORE ANYONE ELSE DOES.
    #
    #  A squashfs block can be written badly and stay invisible: the header,
    #  the directory table and every small file read perfectly, so
    #  build-versions.json prints, python imports work, `ls` shows all four
    #  safetensors at the right size, and the image looks finished. The fault
    #  only appears when the block is finally inflated -- as
    #      SQUASHFS error: zlib decompression failed, data probably corrupt
    #  and, to the caller, as SIGBUS or EIO at "Loading checkpoint shards: 0%".
    #
    #  That is what happened to the 2026-08-31 mslk build: a bad block 6.8 GB
    #  in. It survived an scp to another machine (sha256 matches, because the
    #  copy is faithful -- to a broken original) and cost three rounds of
    #  debugging on the wrong machine. Reading 16 GB back here takes a couple
    #  of minutes and turns that into a build failure.
    # ------------------------------------------------------------------ #
    info "verifying the image can be read back (this is not optional -- see comment)"
    if "$SING" exec "$LLM_OUT" bash -lc 'set -e; d=/opt/model/retrodfm-r-8b; cat "$d"/* > /dev/null'; then
      info "  all model files decompress and read OK"
      if command -v sha256sum >/dev/null 2>&1; then
        info "  sha256: $(sha256sum "$LLM_OUT" | cut -d" " -f1)"
        info "  (match this on the target machine AFTER dropping caches)"
      fi
    else
      echo
      echo "BUILD FAILED VERIFICATION: $LLM_OUT cannot be read back." >&2
      echo "  A block failed to decompress. Check 'sudo dmesg | tail' for" >&2
      echo "  'SQUASHFS error'. Do NOT ship this image -- rebuild it." >&2
      echo "  Free disk space first; a build on a nearly-full filesystem is" >&2
      echo "  the usual cause." >&2
      exit 1
    fi
  fi
}

build_llm

rm -rf "$STAGE"
say "DONE"
cat <<EOF
   The image is self-contained. Put it in run/, next to qb.sh and the other
   files, on the machine with the GPU (it needs a GPU driver and singularity,
   nothing else):

       cp $LLM_OUT $ROOT/run/

   Then, inside run/:

       bash check_sif.sh --deep                                     # image vs card
       bash qb.sh --levels bf16 --n 5 --gpus 1                      # smoke test
       bash qb.sh --levels bf16,int8_ao,nf4,fp4,nf4dq --gpus 4

   qb.sh finds the .sif beside it whatever it is called and needs no paths.
   To read back the exact environment:

       singularity exec $LLM_OUT cat /opt/qb/build-versions.json
EOF
