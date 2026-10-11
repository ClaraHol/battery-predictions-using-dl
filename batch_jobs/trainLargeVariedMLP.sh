#BSUB -J TrainLargeMLP
#BSUB -o outputs/TrainLargeVariedMLP.out
#BSUB -e outputs/TrainLargeVariedMLP.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 05:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

module load python3/3.13.11
source .venv/bin/activate



python3 -u project/src/models/train_Large_varied_MLP_resumable.py --max-epochs 200
