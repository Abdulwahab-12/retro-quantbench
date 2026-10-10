#!/usr/bin/env bash
# ===========================================================================
#  How much GPU memory does each quantization level REALLY need?
#
#      bash qb_memcheck.sh
#
#  That is the whole command. Run it in the folder holding qb.sh and the .sif;
#  like qb.sh, the folder defaults to wherever this script lives.
#
#  ONE MOLECULE PER LEVEL. Peak memory is reached inside the first molecule and
#  does not grow with n, so n=1 gives the same answer as n=5005 in minutes
#  instead of days. No accuracy or speed run needs repeating.
#
#  Two passes per level:
#      1. default model      1 x 1 x 1
#      2. augmented model   20 x 10 x 10, UN-CHUNKED
#
#  WHY UN-CHUNKED. --ans-chunk lowers the peak, which is the point of the flag
#  -- but this script reports what the PUBLISHED configuration needs. If a
#  configuration does not fit a 12 GB card, the report must SHOW a peak above
#  12 GB rather than hide it. On a card too small for it the level fails; that
#  failure is the result, the peak reached up to that point is still recorded,
#  and the script carries on to the next level.
#
#  Options, none of them required:
#      QB_LEVELS=nf4,fp4      which levels   (default: the five in Table 1)
#      QB_KA/QB_KS/QB_KB      other budget   (default 20/10/10)
#      QB_AC=2                extra pass WITH --ans-chunk 2, to show the saving
#      QB_FAST=1              k_a=1 for pass 2: same memory pressure, ~20x
#                             quicker, peak ~2% low (measured 6.76 vs 6.92 GB)
# ===========================================================================
set -u

# Folder defaults to this script's own directory. A directory may still be
# given as the first argument, exactly as qb.sh allows.
if [ "$#" -ge 1 ] && [ -d "$1" ]; then
  RUN="$(cd "$1" && pwd)"
else
  RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
LEVELS="${QB_LEVELS:-bf16,int8_ao,nf4,fp4,nf4dq}"
KA="${QB_KA:-20}"; KS="${QB_KS:-10}"; KB="${QB_KB:-10}"
[ "${QB_FAST:-0}" = 1 ] && KA=1

[ -f "$RUN/qb.sh" ] || { echo "qb.sh is not in $RUN -- run this from the folder holding the .sif"; exit 1; }
N=0; for c in "$RUN"/*.sif; do [ -f "$c" ] && N=$((N+1)); done
[ "$N" -eq 1 ] || { echo "$N .sif files in $RUN -- qb.sh needs exactly one"; exit 1; }
# The worker beside qb.sh replaces the image's own copy, so it must have the
# driver-peak sampler.
for W in "$RUN/qb-run_llm.py" "$RUN/run_llm.py"; do
  if [ -f "$W" ] && ! grep -q "driver_peak_gb" "$W"; then
    echo "$W has no driver-peak sampler; use qb-run_llm.py from this repository."
    exit 1
  fi
done

echo "==========================================================="
echo " folder : $RUN"
echo " levels : $LEVELS"
echo " budget : ${KA} x ${KS} x ${KB}, un-chunked (published configuration)"
echo " one molecule per level; expect a few minutes per level"
echo "==========================================================="
echo
echo "########## pass 1 of 2: default model, 1 x 1 x 1 ##########"
bash "$RUN/qb.sh" --levels "$LEVELS" --n 1 --gpus 1 --ka 1 --ks 1 --kb 1 || true
echo
echo "########## pass 2 of 2: augmented model, ${KA} x ${KS} x ${KB} ##########"
echo "   a level that fails here does not fit this card -- that IS the result"
bash "$RUN/qb.sh" --levels "$LEVELS" --n 1 --gpus 1 --ka "$KA" --ks "$KS" --kb "$KB" || true
if [ "${QB_AC:-0}" != 0 ]; then
  echo
  echo "########## extra pass: --ans-chunk ${QB_AC}, what chunking saves ##########"
  bash "$RUN/qb.sh" --levels "$LEVELS" --n 1 --gpus 1 --ka "$KA" --ks "$KS" --kb "$KB" \
       --ans-chunk "$QB_AC" || true
fi

echo
echo "########################## MEMORY TABLE ##########################"
printf '%-10s %-12s %-9s %8s %11s %10s %7s\n' \
       level setting ans-chunk live peak_alloc peak_REAL card
find "$RUN/results" -name '*.meta.json' 2>/dev/null | sort | while read -r f; do
  python3 - "$f" <<'PY' 2>/dev/null
import json,sys
m=json.load(open(sys.argv[1]))
print("%-10s %-12s %-9s %8s %11s %10s %7s" % (
    m.get("quant","?"),
    "%sx%sx%s" % (m.get("k_a","?"), m.get("k_s","?"), m.get("k_b","?")),
    m.get("ans_chunk","-"), m.get("resident_gb","-"),
    m.get("gen_peak_gb","-"), m.get("driver_peak_gb","-"),
    m.get("device_total_gb","-")))
PY
done
echo
echo "  live       tensors PyTorch is holding"
echo "  peak_alloc PyTorch's own peak -- what the paper reports today"
echo "  peak_REAL  what the DRIVER reports on the card. THIS is the number to"
echo "             compare against a card's capacity: it includes the CUDA"
echo "             context and library workspaces that PyTorch does not count."
echo "             On a shared card it also counts other processes."
echo
echo "  A level missing from pass 2 did not fit; see results/logs/ for its error."
