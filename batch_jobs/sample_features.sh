#BSUB -J SampleApproximateFeatures
#BSUB -o outputs/SampleFeatures.out
#BSUB -e outputs/SampleFeatures.err
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 2GB

module load python3/3.13.11
source .venv/bin/activate



python3 -u project/src/simulation_experiments/sample_approximate_designs.py
