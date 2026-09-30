#BSUB -J EvalModels[1-9]
#BSUB -o outputs/EvalModels_%J_%I.out
#BSUB -e outputs/EvalModels_%J_%I.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 00:10
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB

module load python3/3.13.11
source .venv/bin/activate

CHEMISTRIES=(25R HG2 M1A MJ1 PA PBC PD VTC5A VTC6)

CHEM=${CHEMISTRIES[$((LSB_JOBINDEX - 1))]}

python project/src/models/evaluate_models.py --chemistry "$CHEM"
