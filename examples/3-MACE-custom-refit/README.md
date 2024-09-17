# MACE: fine-tuning the model when needed

Same as `2-MACE-refit`, but the refitting script is provided locally.
You can write your own, and provide the python import path to it:
```yaml
...
refit:
  function_name: "custom_refit.refit_mace.refit_generic"
...
```

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