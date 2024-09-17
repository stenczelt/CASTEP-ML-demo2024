# GAP in CASTEP demo 2024/09/17

Getting started on Mahti

Either go to https://www.mahti.csc.fi/ and log in through the browser, or

start an interactive session
```bash
sinteractive --account project_2010950 --time 24:00:00 --cores 4
```

copy the demo files
```bash
cp -r /projappl/project_2010950/castep-ml/copy-this/ castep-ml-demo
cd castep-ml-demo
```

Further reading: 
- documentation: https://libatoms.github.io/GAP/accelerated-aimd.html

## Structure

`code/` contains a copy of `mace` & `hybrid-md` tailored to our setup

`examples/` contains a series of examples:
- `0-GAP-from-scratch` GAP with TurboSOAP trained from scratch
- `1-MACE-only-checking` MACE-MP0 with only checking accuracy
- `2-MACE-refit` fine-tuning MACE-MP0 on-the-fly
- `3-MACE-custom-refit` fine-tuning with user-supplied logic for refitting
- `4-try-your-own` try this on your own system

## Getting started:

load modules & paths
```
module use --append /projappl/project_2010950/modules
module load castep/cpu+QUIP+MACE
```

create a python environment
```
cd examples/
bash bootstrap.sh
source venv/bin/activate
```

or use `/projappl/project_2010950/castep-ml/venv`

Can run each 

