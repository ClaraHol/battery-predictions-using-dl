#BSUB -J TrainLargeNSF
#BSUB -o outputs/TrainLargeNSF.out
#BSUB -e outputs/TrainLargeNSF.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 04:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

module load python3/3.13.11
source .venv/bin/activate



python project/src/models/train_Large_NSF.py
