#! /bin/bash

#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<youremail@gmail.com>
#SBATCH --mail-type=ALL

cd ~/$PROJECT/ComputeCanadaSample
module purge
module load python/3.12.4 scipy-stack
source ~/venv/bin/activate

python main.py
