#!/bin/bash

module use --append /projappl/project_2010950/modules
module load castep/cpu+QUIP+MACE

python -m venv venv
source ./venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ../code/mace/
pip install -e ../code/hybrid_md_package/
