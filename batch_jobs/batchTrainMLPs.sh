#BSUB -J TrainChem[1-9]
#BSUB -o outputs/TrainMLP_%J_%I.out
#BSUB -e outputs/TrainMLP_%J_%I.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 04:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

module load python3/3.13.11
source .venv/bin/activate

CHEMISTRIES=(25R HG2 M1A MJ1 PA PBC PD VTC5A VTC6)
CHEM=${CHEMISTRIES[$((LSB_JOBINDEX - 1))]}

python project/src/models/train_MLPs.py --chemistry "$CHEM"
