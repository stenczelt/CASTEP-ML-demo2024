# MACE: fine-tuning the model when needed

Use MACE-MP0 (no need to copy now), run 3 steps of AIMD for initial data, and fine-tune
the model to the current DFT results.

load the env
```
module use --append /projappl/project_2010950/modules
module load castep/cpu+QUIP+MACE
source ../venv/bin/activate
```

run the calculation - should be reasonable on 1-2 cores 
```
castep.mpi sic
```

Note, this will be slow on CPU.

