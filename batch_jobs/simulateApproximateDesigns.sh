```bash
#!/bin/bash
#BSUB -J "ApproxSim[1-9]"
#BSUB -o outputs/ApproxSim_%J_%I.out
#BSUB -e outputs/ApproxSim_%J_%I.err
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB

set -euo pipefail

module load python3/3.13.11
source .venv/bin/activate

CHEMISTRIES=(25R HG2 M1A MJ1 PA PBC PD VTC5A VTC6)
CHEM=${CHEMISTRIES[$((LSB_JOBINDEX - 1))]}

# Avoid thread oversubscription inside each multiprocessing worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u project/src/simulation_experiments/simulate_approximate_design.py \
    --chemistry "$CHEM" \
    --accuracy realistic \
    --num-successful 100000 \
    --chunk-size 1000 \
    --num-workers 8 \
    --max-attempts-per-design 3 \
    --seed 0 \
    --overwrite
```