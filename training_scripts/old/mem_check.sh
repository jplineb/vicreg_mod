#!/bin/bash

#PBS -N CheX_LPM
#PBS -l select=1:ncpus=56:mem=275gb:ngpus=1:gpu_model=v100,walltime=10:00:00
#PBS -M jplineb@clemson.edu
#PBS -j oe

epochs=5
lr_backbone=0.0
lr_head=0.03
batch_size=120
weights='freeze'

echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Memory Usage Analysis"
echo "----------------------------"

# Load modules
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0
module load cudnn/8.7.0.84-11.8-gcc/9.5.0

# /software/ModuleFiles/modules/linux-rocky8-x86_64/cudnn/8.0.5.39-11.1-gcc/9.5.0-cu11_1
# Activate Conda and CD
source activate pda
cd /home/jplineb/VICReg/vicreg_mod/

# Linear Probe on Shallow RestNet50 RadImagenet
memray run --trace-python-allocators --native --follow-fork evaluate_chexpert.py \
    --pretrained_path /zfs/wficai/radimagenet/bigModel/checkpoint.pth \
    --exp-dir ./checkpoint/chexpert_ResNet50_RadImageNet_LP \
    --pretrained-how Shallow \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    # --resume