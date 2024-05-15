# This script is meant to be run in the command line, not submitted via sbatch

source /usr/local/linux/mambaforge-3.11/bin/activate ucsf_env
cd /accounts/campus/austin.zane/ucsf_fast

python \
    -m experiments.train_segmentation \
    --model_saving_off \
