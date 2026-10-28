#!/bin/bash
#BSUB -J "ApproxDesign[1-9]"
#BSUB -o outputs/ApproxDesign_%J_%I.out
#BSUB -e outputs/ApproxDesign_%J_%I.err
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -W 02:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB

set -euo pipefail

module load python3/3.13.11
source .venv/bin/activate

CHEMISTRIES=(25R HG2 M1A MJ1 PA PBC PD VTC5A VTC6)
CHEM=${CHEMISTRIES[$((LSB_JOBINDEX - 1))]}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u project/src/simulation_experiments/sample_approximate_designs_batch.py \
    --chemistry "$CHEM" \
    --n 200000 \
    --accuracy realistic \
    --seed 0 \
    --min-inactive-fraction 0.02 \
    --capacity-source used \
    --checkpoint-every 10000 \
    --resume
