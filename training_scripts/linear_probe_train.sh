#!/bin/bash
#PBS -N CheX_LP3
#PBS -l select=1:ncpus=56:mem=250gb:ngpus=1:gpu_model=v100,walltime=08:00:00
#PBS -M jplineb@clemson.edu
#PBS -j oe

epochs=5
lr_backbone=0.0
lr_head=0.03
batch_size=110
weights='freeze'

# training=$1
training="VRRIM"

echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Linear Probe Experiments"
echo "----------------------------"

# Load modules
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0

# Activate Conda and CD
source activate pda
cd /home/jplineb/VICReg/vicreg_mod/

if [ $training == "VRIM" ]
then
    # Linear Probe on VICReg ImageNet
    python evaluate_chexpert.py \
        --pretrained_path ./VICReg_ImageNet/resnet50.pth \
        --exp-dir ./checkpoint/chexpert_VICReg_ImageNet_LP \
        --pretrained-how VICReg \
        --pretrained-dataset ImageNet \
        --epochs $epochs \
        --weights $weights \
        --lr-backbone $lr_backbone \
        --lr-head $lr_head \
        --batch-size $batch_size \
        --resume
elif [ $training == "VRRIM" ]:
then
    # Linear Probe on VICReg RadImageNet
    python evaluate_chexpert.py \
        --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
        --exp-dir ./checkpoint/chexpert_VICReg_RadImageNet_LP \
        --pretrained-how VICReg \
        --pretrained-dataset RadImageNet \
        --epochs $epochs \
        --weights $weights \
        --lr-backbone $lr_backbone \
        --lr-head $lr_head \
        --batch-size $batch_size \
        # --resume
elif [ $training == "SIM" ]
then
    # Linear Probe on Shallow ResNet50 ImageNet
    python evaluate_chexpert.py \
        --pretrained_path ./ \
        --exp-dir ./checkpoint/chexpert_ResNet50_ImageNet_LP \
        --pretrained-how Shallow \
        --pretrained-dataset ImageNet \
        --epochs $epochs \
        --weights $weights \
        --lr-backbone $lr_backbone \
        --lr-head $lr_head \
        --batch-size $batch_size \
        --resume
elif [ $training == "SRIM" ]
then
    # Linear Probe on Shallow RestNet50 RadImagenet
    python evaluate_chexpert.py \
        --pretrained_path /zfs/wficai/radimagenet/bigModel/checkpoint.pth \
        --exp-dir ./checkpoint/chexpert_ResNet50_RadImageNet_LP \
        --pretrained-how Shallow \
        --pretrained-dataset RadImageNet \
        --epochs $epochs \
        --weights $weights \
        --lr-backbone $lr_backbone \
        --lr-head $lr_head \
        --batch-size $batch_size \
        --resume
else
    echo "No training parameter passed"
fi