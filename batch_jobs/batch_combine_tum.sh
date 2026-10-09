#BSUB -J Combine_tum
#BSUB -o outputs/Combine_tum_%J_%I.out
#BSUB -e outputs/Combine_tum_%J_%I.err
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 01:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

module load python3/3.13.11
source .venv/bin/activate

cd /zhome/07/c/168354/courses/battery-predictions-using-dl/project/src/data/data_processing

python3 -u tum_processing.py /work3/claho/battery_datasets/sony_vtc5a /work3/claho/battery_datasets/sony_vtc5a/bawaii_cell_overview_publication.xlsx /work3/claho/battery_datasets/sony_vtc5a/processed_


