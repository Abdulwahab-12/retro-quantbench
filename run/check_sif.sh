#!/usr/bin/env bash
# ===========================================================================
#  Is THIS image right for THIS card?  Ten seconds, no model load.
#
#      bash check_sif.sh
#
#  Run it in the folder holding the .sif. An image that does not match the
#  card fails in several ways -- missing bitsandbytes, "no kernel image is
#  available", sm_120 warnings -- and none of those messages names the cause.
#
#  Send the output as-is. It is short and it is the whole diagnosis.
# ===========================================================================
set -u

SELF="$(basename "${BASH_SOURCE[0]}")"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF=""
DEEP=0

# A .sif may be named directly. qb.sh refuses a folder holding two images and
# should keep refusing -- silently running the wrong one is the exact failure
# this whole script exists to catch. But this script only LOOKS at an image, so
# naming one must be allowed, and "check them one at a time" was useless advice
# without saying how.
for a in "$@"; do
  case "$a" in
    --deep) DEEP=1 ;;
    *)
      if   [ -f "$a" ]; then SIF="$(cd "$(dirname "$a")" && pwd)/$(basename "$a")"
      elif [ -d "$a" ]; then D="$(cd "$a" && pwd)"
      fi ;;
  esac
done

if [ -z "$SIF" ]; then
  N=0
  for c in "$D"/*.sif; do [ -f "$c" ] || continue; N=$((N+1)); SIF="$c"; done
  if [ "$N" -eq 0 ]; then echo "no .sif found in $D" >&2; exit 1; fi
  if [ "$N" -gt 1 ]; then
    echo "more than one .sif in $D -- name the one to check:" >&2
    for c in "$D"/*.sif; do echo "    bash $SELF $c" >&2; done
    exit 1
  fi
fi

SING="$(command -v singularity || command -v apptainer || true)"
[ -n "$SING" ] || { echo "singularity/apptainer not installed" >&2; exit 1; }

REAL="$(readlink -f "$SIF" 2>/dev/null || echo "$SIF")"
echo "=========================================================="
echo "IMAGE"
echo "=========================================================="
echo "  path   : $SIF"
[ "$REAL" != "$SIF" ] && echo "  -> real: $REAL"
echo "  size   : $(du -h "$REAL" 2>/dev/null | cut -f1)   built $(date -r "$REAL" '+%Y-%m-%d %H:%M' 2>/dev/null)"
echo "  bytes  : $(stat -c %s "$REAL" 2>/dev/null)   <- must match the source machine exactly"

# WHERE the image lives decides whether it is all there. A 19 GB image copied
# onto a filesystem with less than 19 GB free is silently truncated: ls shows
# the right sizes (squashfs metadata sits near the start) and reading the
# weights fails with EIO or SIGBUS deep inside. /tmp is the usual trap -- often
# small, often tmpfs (that is RAM), and cleaned without warning.
FS_TYPE="$(stat -f -c %T "$REAL" 2>/dev/null || echo unknown)"
FS_AVAIL="$(df -h --output=avail "$REAL" 2>/dev/null | tail -1 | tr -d ' ')"
FS_MOUNT="$(df --output=target "$REAL" 2>/dev/null | tail -1)"
echo "  on     : $FS_MOUNT  ($FS_TYPE, $FS_AVAIL free)"
case "$REAL" in
  /tmp/*) echo "           WARNING: /tmp is usually small and is cleared without notice."
          echo "           Keep the image somewhere permanent with >25 GB free." ;;
esac
case "$FS_TYPE" in
  tmpfs|ramfs) echo "           WARNING: $FS_TYPE is RAM. A 19 GB image here consumes 19 GB"
               echo "           of memory and cannot survive a reboot." ;;
esac
echo
"$SING" exec "$SIF" cat /opt/qb/build-versions.json 2>/dev/null \
  | sed 's/^/  /' || echo "  (no build-versions.json -- image predates it)"

# build-versions.json records "torch_cuda_build", which is the CUDA that the
# PyTorch WHEEL was compiled against -- not the CUDA toolkit in the base layer.
# Those are different numbers and the JSON only shows one, which reads as though
# the requested base version had been ignored. Show both, side by side.
BASE="$("$SING" exec "$SIF" bash -lc \
        'nvcc --version 2>/dev/null | grep -oE "release [0-9]+\.[0-9]+" | head -1' \
        2>/dev/null)"
echo
echo "  base CUDA toolkit in the image : ${BASE:-unknown}"
echo "  CUDA the torch wheel was built for: see torch_cuda_build above"
echo "  These differ ON PURPOSE. PyTorch publishes no cu133 wheel -- the newest"
echo "  channel is cu132 -- and MSLK 1.3.0 ships only +cu130 and +cu132. The"
echo "  wheels carry their own CUDA runtime, so the base version barely matters."

# WSL needs the driver libraries bound in, or torch reports no GPU at all and
# this script would blame the image for a binding problem.
BINDS=()
if [ -d /usr/lib/wsl ]; then
  BINDS+=(--bind /usr/lib/wsl:/usr/lib/wsl)
  if "$SING" --version 2>/dev/null | grep -qi apptainer; then
    export APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/wsl/lib"
  else
    export SINGULARITYENV_LD_LIBRARY_PATH="/usr/lib/wsl/lib"
  fi
fi

"$SING" exec --nv ${BINDS+"${BINDS[@]}"} "$SIF" python - <<'PY'
import sys, importlib

def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "present")
    except Exception as e:
        return f"ABSENT ({type(e).__name__})"

print("=" * 58)
print("SOFTWARE IN THE IMAGE")
print("=" * 58)
print(f"  python       : {sys.version.split()[0]}")
import torch
print(f"  torch        : {torch.__version__}   built for CUDA {torch.version.cuda}")
print(f"  torchao      : {ver('torchao')}")
print(f"  bitsandbytes : {ver('bitsandbytes')}      <- nf4 / nf4dq / fp4 need this")
print(f"  mslk         : {ver('mslk')}")
print(f"  transformers : {ver('transformers')}")

print()
print("=" * 58)
print("THE CARD, AND WHETHER THIS IMAGE CAN DRIVE IT")
print("=" * 58)
if not torch.cuda.is_available():
    print("  torch sees NO GPU.")
    print("  On WSL this is usually the driver bind, not the image.")
    sys.exit(1)

arch = list(torch.cuda.get_arch_list() or [])
print(f"  this torch was built for: {' '.join(arch) or '(unknown)'}")
print()
bad = False
for i in range(torch.cuda.device_count()):
    cc = torch.cuda.get_device_capability(i)
    tag = f"sm_{cc[0]}{cc[1]}"
    name = torch.cuda.get_device_name(i)
    gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
    ok = (not arch) or (tag in arch)
    bad |= not ok
    print(f"  gpu {i}: {name}  {gb:.0f} GB  {tag}   "
          + ("OK" if ok else "NOT SUPPORTED BY THIS IMAGE"))

print()
print("=" * 58)
print("VERDICT")
print("=" * 58)
if bad:
    print("  WRONG IMAGE FOR THIS MACHINE.")
    print("  Every level will fail, each with a different error message, and")
    print("  none of them will say this. Nothing produced would be a result.")
    print()
    # Advise by the CARD. The images built by docker/build_images.sh use
    # CUDA 13, which targets sm_75 (Turing) and newer.
    if min(torch.cuda.get_device_capability(i)
           for i in range(torch.cuda.device_count())) < (7, 5):
        print("  This card is older than sm_75 (Turing). The images built by")
        print("  docker/build_images.sh use CUDA 13 and cannot drive it.")
    else:
        print("  Build the image with:  bash docker/build_images.sh --mslk")
        print("  It targets sm_75 to sm_120.")
    sys.exit(1)

print("  This image can drive this card.")

try:
    import bitsandbytes  # noqa: F401
except Exception:
    print()
    print("  BUT bitsandbytes is missing, so nf4, nf4dq and fp4 cannot run.")
    print("  Everything else can. This is the package, not the card: rebuild")
    print("  with docker/build_images.sh --mslk.")
    sys.exit(2)
print("  bitsandbytes present, so the nf4 / fp4 levels can run too.")
print()
# Recommend a level that FITS. bf16 weights are ~16 GB, so on an 11 GB card
# "bash qb.sh --levels bf16 --n 4" is guaranteed to die during loading -- which
# it duly did on a 2080 Ti, after a clean probe, and read as a failure of the
# image rather than of the advice.
_min_gb = min(torch.cuda.get_device_properties(i).total_memory / 1024**3
              for i in range(torch.cuda.device_count()))
if _min_gb >= 20:
    print("  Next:  bash qb.sh --levels bf16 --n 4        (a real 4-molecule test)")
else:
    print(f"  This card has {_min_gb:.0f} GB. bf16 weights are ~16 GB and will NOT")
    print("  load -- use a quantized level instead:")
    print("  Next:  bash qb.sh --levels nf4 --n 4         (a real 4-molecule test)")
PY

# --- deep read: is the image itself intact? --------------------------------
# A .sif that is still copying, or was cut short, reads perfectly near the
# start -- build-versions.json, the python imports, all fine -- and fails deep
# inside, which is exactly where the 16 GB of weights sit. That surfaces as
# SIGBUS (signal 7) at "Loading checkpoint shards: 0%" and nowhere else, so
# every level fails identically and none of them says why.
if [ "$DEEP" = 1 ]; then
  echo
  echo "=========================================================="
  echo "DEEP READ -- proving the image is not truncated"
  echo "=========================================================="
  echo "  host-side size: $(stat -c %s "$REAL" 2>/dev/null) bytes"
  echo "  Compare against the machine you copied it FROM:"
  echo "      sha256sum $REAL"
  echo
  echo "  reading every model file inside the image ..."
  if "$SING" exec "$SIF" bash -lc 'set -e; d=/opt/model/retrodfm-r-8b; ls -l "$d"; echo; cat "$d"/* > /dev/null; echo "ALL MODEL FILES READ OK"'; then
    echo
    echo "  Image intact -- SIGBUS was NOT a truncated .sif."
    echo "  Look at storage instead:   df -h / /tmp /dev/shm"
    echo "                             dmesg | tail -30"
  else
    echo
    echo "  READ FAILED. This image is incomplete, or the storage under it is"
    echo "  faulty. Re-copy it and match sha256sum before running again."
    exit 1
  fi
fi
