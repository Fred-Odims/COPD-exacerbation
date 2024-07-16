#!/bin/bash
#SBATCH --job-name=copd_analysis
#SBATCH --cpus-per-task=1   # Allocate 1 CPU core per task.
#SBATCH --mem=4G  # Memory per node
#SBATCH --time=00:10:00  # Time (DD:HH:MM)
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=unfodimba@mun.ca
#SBATCH --mail-type=ALL

#load modules
module load python/3.11.5 scipy-stack/2023b

#Create and activate virtual environment
virtualenv --no-download COPD_ENV
source COPD_ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index scikit_learn==1.3.1 seaborn==0.13.2 imblearn

# launch the python script
python copd_jobscript.py
