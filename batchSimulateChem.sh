#BSUB -J SimulateChem[1-8]
#BSUB -o outputs/SimulateChem_%J_%I.out
#BSUB -e outputs/SimulateChem_%J_%I.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 06:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

module load python3/3.13.11
source .venv/bin/activate

CHEMISTRIES=(25R HG2 M1A MJ1 PA PBC PD VTC5A VTC6)
CHEM=${CHEMISTRIES[$((LSB_JOBINDEX - 1))]}

python project/src/simulation/SBI_construction/generate_simulations.py --chemistry "$CHEM"