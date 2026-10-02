#!/usr/bin/env bash
# ===========================================================================
#  WHY IS THE AUGMENTED MODEL ONLY ~60x SLOWER WHEN IT MAKES 2000 PREDICTIONS?
#
#      bash qb_speedmath.sh
#
#  Settles it with every term MEASURED on one card, one level, no assumptions.
#  Runs the same molecules at 1x1x1 and at 20x10x10 and checks that
#
#         time ratio  =  token ratio  /  throughput ratio
#
#  Each record already stores n_gen_tokens and seconds, so all three ratios
#  come straight out of the result files.
#
#  Options:
#      QB_LEVELS=bf16,nf4   default. Add more at your own cost.
#      QB_N1=10             molecules for the 1x1x1 pass (cheap)
#      QB_N2=1              molecules for the 20x10x10 pass (expensive)
#      QB_GPUS=1            keep at 1: throughput per CARD is the quantity
#                           being measured, and more workers do not change it
#                           but do make s/mol ambiguous.
#
#  TIME: the 1x1x1 pass is minutes. The 20x10x10 pass takes 6-9 minutes per
#  molecule and level on an RTX 5090. Run it with nohup and come back.
# ===========================================================================
set -u
if [ "$#" -ge 1 ] && [ -d "$1" ]; then RUN="$(cd "$1" && pwd)"
else RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
LEVELS="${QB_LEVELS:-bf16,nf4}"
N1="${QB_N1:-10}"; N2="${QB_N2:-1}"; G="${QB_GPUS:-1}"

[ -f "$RUN/qb.sh" ] || { echo "qb.sh is not in $RUN"; exit 1; }
N=0; for c in "$RUN"/*.sif; do [ -f "$c" ] && N=$((N+1)); done
[ "$N" -eq 1 ] || { echo "$N .sif files in $RUN -- qb.sh needs exactly one"; exit 1; }

echo "folder : $RUN"
echo "levels : $LEVELS      gpus: $G"
echo "pass 1 : 1x1x1     n=$N1   (minutes)"
echo "pass 2 : 20x10x10  n=$N2   (6-9 min per molecule and level on an RTX 5090)"
echo
echo "############ pass 1: un-augmented, 1 x 1 x 1 ############"
bash "$RUN/qb.sh" --levels "$LEVELS" --n "$N1" --gpus "$G" --ka 1 --ks 1 --kb 1 || true
echo
echo "############ pass 2: augmented, 20 x 10 x 10 ############"
bash "$RUN/qb.sh" --levels "$LEVELS" --n "$N2" --gpus "$G" --ka 20 --ks 10 --kb 10 || true
echo
bash "$RUN/qb_speedmath_report.sh" "$RUN"
