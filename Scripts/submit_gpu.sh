#!/bin/ksh
# **********************************************************************
# Submission on an available node in the general GPU queue
#$ -q gpu
#$ -pe smp 1               # SMP parallel environment for 1 GPU
#$ -N train                 # The name of the job (optional)
# -j y                          # Join stdout and stderr (optional)
#$ -cwd                         # Use the current working dir (optional)

# **********************************************************************
echo STARTED on $(date)
module load pytorch/2.2.2/gpu 
module list

# Permissions and accesses
# export PATH=/usr/local/cuda-11.5/bin:$PATH # /usr/local/cuda-11.5 is not found (admin permission?)
export PYTHONIOENCODING=utf-8

which python 
export PYTHONPATH="$PYTHONPATH:/user1/icmub/gu9875le/.local/lib/python3.11/site-packages"
cd $SGE_O_WORKDIR

# rm -r logs
# rm -r models 

python train_CNN_LR_batch.py



