# MACE: only checking error

Use MACE-MP0 & 
We are running MD with checking errors every N (10 for the demo) steps.

load the env & copy model 
```
# env
module use --append /projappl/project_2010950/modules
module load castep/cpu+QUIP+MACE
source ../venv/bin/activate

# MACE model from common path, it's not being updated now
cp /projappl/project_2010950/castep-ml/mace-cache/mace_agnesi_mediummodel-lammps.pt MACE-jit.pt
```

run the calculation - should be reasonable on 1-2 cores 
```
castep.mpi sic
```