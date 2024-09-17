#!/bin/bash
#SBATCH --job-name=castep-GAP
#SBATCH --account=project_2010950
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128

module use --append /projappl/project_2010950/modules
module load castep/cpu+QUIP+MACE

source ../venv/bin/activate

srun castep.mpi sic_md