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


filein='./1000-200/valid_rot.p'
fileout='./1000-200/valid_rot_augmented.p'
python augment_validation_rotations.py $filein $fileout
fileout='./1000-200/valid_rot_not_augmented.p'
python augment_validation_rotations.py $filein $fileout --augmentation none



filein='./5k-1k/valid_rot.p'
fileout='./5k-1k/valid_rot_augmented.p'
python augment_validation_rotations.py $filein $fileout
fileout='./5k-1k/valid_rot_not_augmented.p'
python augment_validation_rotations.py $filein $fileout --augmentation none


filein='./10k-2k/valid_rot.p'
fileout='./10k-2k/valid_rot_augmented.p'
python augment_validation_rotations.py $filein $fileout
fileout='./10k-2k/valid_rot_not_augmented.p'
python augment_validation_rotations.py $filein $fileout --augmentation none


filein='./120k-25k/valid_rot.p'
fileout='./120k-25k/valid_rot_augmented.p'
python augment_validation_rotations.py $filein $fileout
fileout='./120k-25k/valid_rot_not_augmented.p'
python augment_validation_rotations.py $filein $fileout --augmentation none


