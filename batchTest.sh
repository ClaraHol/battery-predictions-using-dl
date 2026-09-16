#BSUB -J TrainModel
#BSUB -o outputs/TestingTrainer_%J.out
#BSUB -e outputs/TestingTrainer_%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 04:00
#BSUB -R "rusage[mem=4GB]"

module load python3/3.13.11
source .venv/bin/activate

python project/src/simulation/SBI_construction/generate_model_25R.py 