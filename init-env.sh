# init the environment
alias ll="ls -ltr --color=auto"
module load netlib-scalapack fftw python-data
export PATH="/projappl/project_2008476/tks32/bin:$PATH"
python -m virtualenv venv
source ./venv/bin/activate
python -m pip install ./hybrid_md_package
