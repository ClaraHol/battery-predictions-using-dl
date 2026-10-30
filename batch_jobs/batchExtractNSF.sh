#!/bin/bash
#BSUB -J ExtractNSF[1-5]
#BSUB -o outputs/ExtractNSF_%J_%I.out
#BSUB -e outputs/ExtractNSF_%J_%I.err
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 01:00
#BSUB -R "rusage[mem=3GB]"
#BSUB -M 3GB

module load python3/3.13.11
source .venv/bin/activate

SOURCES=(vtc5a vtc6 samsung_mj1 lg_mj1 samsung_25r)
SOURCE=${SOURCES[$((LSB_JOBINDEX - 1))]}

# Fast diagnostic experiment: direct samples from the learned NSF flow.
# No MCMC and no rejection sampling are used.
python project/src/data/data_processing2/extract_features_nsfv2.py \
    --source "$SOURCE" \
    --n-posterior-samples 5000 \
    --observation-batch-size 16
