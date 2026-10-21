#BSUB -J extract_simple_features
#BSUB -o outputs/extract_simple_features%J.out
#BSUB -e outputs/extract_simple_features%J.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 00:40
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 2GB

module load python3/3.13.11
source .venv/bin/activate


python3 -u project/src/oracle/extract_features_simple.py 