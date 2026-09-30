#BSUB -J Process_cells
#BSUB -o outputs/Process_cells_%J_%I.out
#BSUB -e outputs/Process_cells_%J_%I.err
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 06:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

module load python3/3.13.11
source .venv/bin/activate

cd /zhome/07/c/168354/courses/battery-predictions-using-dl/project/src/data/data_processing

python3 build_public_charging_dataset.py


