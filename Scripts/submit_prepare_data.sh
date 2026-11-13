#!/bin/ksh
#$ -N grid
#$ -pe smp 32
#$ -l vendor=amd
#$ -q batch


echo STARTED on $(date)
module load pytorch/2.2.2/gpu 
module list

# Permissions and accesses
# export PATH=/usr/local/cuda-11.5/bin:$PATH # /usr/local/cuda-11.5 is not found (admin permission?)
export PYTHONIOENCODING=utf-8

which python 
export PYTHONPATH="$PYTHONPATH:/user1/icmub/gu9875le/.local/lib/python3.11/site-packages"
cd $SGE_O_WORKDIR


filein='/beegfs/data/work/icmub/gu9875le/ML/New/Data/Bulk/Bulk_nooutliers/10k-2k/test.p'
fileout='test_grid_small.p'
python create_electrostatic_grid.py $filein $fileout

filein='/beegfs/data/work/icmub/gu9875le/ML/New/Data/Bulk/Bulk_nooutliers/10k-2k/train.p'
fileout='train_grid_small.p'
python create_electrostatic_grid.py $filein $fileout

filein='/beegfs/data/work/icmub/gu9875le/ML/New/Data/Bulk/Bulk_nooutliers/10k-2k/valid.p'
fileout='valid_grid_small.p'
python create_electrostatic_grid.py $filein $fileout

qsub submit_prepare_data.sh

