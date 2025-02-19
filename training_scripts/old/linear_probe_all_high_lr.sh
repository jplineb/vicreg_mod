#!/bin/bash
#PBS -N CheX_LPA
#PBS -l select=1:ncpus=56:mem=372gb:ngpus=1:gpu_model=a100,walltime=10:00:00
#PBS -M jplineb@clemson.edu
#PBS -j oe

epochs=10
lr_backbone=0.0
lr_head=0.1
batch_size=175
weights='freeze'

echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Linear Probe Experiments"
echo "----------------------------"

# jobperf -record -w -rate 10s -http &

# Load modules
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0

# Activate Conda and CD
source activate pda
cd /home/jplineb/VICReg/vicreg_mod/

# Linear Probe on VICReg ImageNet
python evaluate_chexpert.py \
    --pretrained_path ./VICReg_ImageNet/resnet50.pth \
    --exp-dir ./checkpoint/chexpert_VICReg_ImageNet_LP_HLR \
    --pretrained-how VICReg \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 56 \
    # --resume

# Linear Probe on VICReg RadImageNet
python evaluate_chexpert.py \
    --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
    --exp-dir ./checkpoint/chexpert_VICReg_RadImageNet_LP_HLR \
    --pretrained-how VICReg \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 56 \
    # --resume

# # Linear Probe on Shallow ResNet50 ImageNet
# python evaluate_chexpert.py \
#     --pretrained_path ./ \
#     --exp-dir ./checkpoint/chexpert_ResNet50_ImageNet_LP \
#     --pretrained-how Supervised \
#     --pretrained-dataset ImageNet \
#     --epochs $epochs \
#     --weights $weights \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers 56 \
#     # --resume

# # Linear Probe on Shallow RestNet50 RadImagenet
# python evaluate_chexpert.py \
#     --pretrained_path /zfs/wficai/radimagenet/bigModel/checkpoint.pth \
#     --exp-dir ./checkpoint/chexpert_ResNet50_RadImageNet_LP \
#     --pretrained-how Supervised \
#     --pretrained-dataset RadImageNet \
#     --epochs $epochs \
#     --weights $weights \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers 56 \
#     # --resume