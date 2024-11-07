#!/bin/bash
#PBS -N CheX_LPA
#PBS -l select=1:ncpus=56:mem=370gb:ngpus=1:gpu_model=a100,walltime=8:00:00
#PBS -M jplineb@clemson.edu
#PBS -j oe

epochs=5
lr_backbone=0.01
lr_head=0.03
batch_size=110
weights='finetune'

echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Linear Probe Experiments"
echo "----------------------------"

jobperf -record -w -rate 10s -http &

# Load modules
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0

# Activate Conda and CD
source activate pda
cd /home/jplineb/VICReg/vicreg_mod/

# # FT on VICReg ImageNet
# python evaluate_mimiccxr.py \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/mimiccxr_VICReg_ImageNet_ft \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs $epochs \
#     --weights $weights \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers 56 \
#     # --resume

# # FT on Shallow RestNet50 RadImagenet
python evaluate_chexpert.py \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/chexpert_ResNet50_RadImageNet_ft \
    --pretrained-how Supervised \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 50 \
    # --resume

# python evaluate_mimiccxr.py \
#     --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
#     --exp-dir ./checkpoint/mimiccxr_ResNet50_RadImageNet_ft \
#     --pretrained-how Supervised \
#     --pretrained-dataset RadImageNet \
#     --epochs $epochs \
#     --weights $weights \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers 50 \
#     # --resume


######
# Linear Probe on VICReg ImageNet
python evaluate_chexpert.py \
    --pretrained_path ./VICReg_ImageNet/resnet50.pth \
    --exp-dir ./checkpoint/chexpert_VICReg_ImageNet_ft \
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
    --exp-dir ./checkpoint/chexpert_VICReg_RadImageNet_ft \
    --pretrained-how VICReg \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 56 \
    # --resume

# python evaluate_mimiccxr.py \
#     --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/mimiccxr_VICReg_RadImageNet_ft \
#     --pretrained-how VICReg \
#     --pretrained-dataset RadImageNet \
#     --epochs $epochs \
#     --weights $weights \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers 50 \
#     # --resume

# # Linear Probe on Shallow ResNet50 ImageNet
python evaluate_chexpert.py \
    --pretrained_path ./ \
    --exp-dir ./checkpoint/chexpert_ResNet50_ImageNet_ft \
    --pretrained-how Supervised \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 56 \
    # --resume

# python evaluate_mimiccxr.py \
#     --pretrained_path ./ \
#     --exp-dir ./checkpoint/mimiccxr_ResNet50_ImageNet_ft \
#     --pretrained-how Supervised \
#     --pretrained-dataset ImageNet \
#     --epochs $epochs \
#     --weights $weights \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers 56 \
#     # --resume

