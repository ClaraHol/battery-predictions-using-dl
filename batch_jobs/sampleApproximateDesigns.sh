#!/bin/bash
#BSUB -J "ApproxDesign[1-9]"
#BSUB -o outputs/ApproxDesign_%J_%I.out
#BSUB -e outputs/ApproxDesign_%J_%I.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 00:20
#BSUB -R "rusage[mem=1GB]"
#BSUB -M 1GB

set -euo pipefail

# Submit from project root, after: mkdir -p outputs
module load python3/3.13.11
source .venv/bin/activate

CHEMISTRIES=(25R HG2 M1A MJ1 PA PBC PD VTC5A VTC6)
CHEM=${CHEMISTRIES[$((LSB_JOBINDEX - 1))]}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u project/src/simulation_experiments/sample_approximate_designs_batch.py \
    --chemistry "$CHEM" \
    --n 500 \
    --accuracy realistic \
    --seed 0 \
    --min-inactive-fraction 0.02 \
    --capacity-source used \
    --checkpoint-every 1000 \
    --overwrite
