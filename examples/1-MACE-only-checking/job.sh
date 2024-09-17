#!/bin/bash
#SBATCH --job-name=castep-MACE-check
#SBATCH --account=project_2010950
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module use --append /projappl/project_2010950/modules
module load castep/cpu+QUIP+MACE

source ../venv/bin/activate

# MACE model from common path, it's not being updated now
cp /projappl/project_2010950/castep-ml/mace-cache/mace_agnesi_mediummodel-lammps.pt MACE-jit.pt

srun castep.mpi sic
