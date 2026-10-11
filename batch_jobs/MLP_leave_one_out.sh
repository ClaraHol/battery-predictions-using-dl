#!/bin/bash
#BSUB -J "MLP_LeaveOneOut[1-9]"
#BSUB -o outputs/MLP_LeaveOneOut_%J_%I.out
#BSUB -e outputs/MLP_LeaveOneOut_%J_%I.err
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -W 03:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

module load python3/3.13.11
source .venv/bin/activate

CHEMISTRIES=(25R HG2 M1A MJ1 PA PBC PD VTC5A VTC6)
CHEM=${CHEMISTRIES[$((LSB_JOBINDEX - 1))]}

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Excluded chemistry: $CHEM"
echo "Host: $(hostname)"
echo "Start: $(date)"

python -u project/src/models/MLP_train_varied_leave_one_out.py \
    --exclude "$CHEM" \
    --max-epochs 300 \
    --resume

echo "Finished: $(date)"