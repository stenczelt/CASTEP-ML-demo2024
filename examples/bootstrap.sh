#!/bin/bash

#  Hybrid MD decision making package
#
#  Copyright (c) Tamas K. Stenczel 2021-2024.

module use --append /projappl/project_2010950/modules
module load castep/cpu+QUIP+MACE

python -m venv venv
source ./venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ../code/mace/
pip install -e ../code/hybrid_md_package/
