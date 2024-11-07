#!/bin/bash
#PBS -N Vindr_cxr
#PBS -l select=1:ncpus=56:mem=370gb:ngpus=1:gpu_model=p100,walltime=8:00:00
#PBS -M jplineb@clemson.edu
#PBS -j oe

epochs=10
lr_backbone=1e-3
lr_head=0.03
batch_size=120
weights='finetune'
weightdecay=1e-4

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

# FT on VICReg ImageNet
python evaluate_vindrcxr.py \
    --pretrained_path ./VICReg_ImageNet/resnet50.pth \
    --exp-dir ./checkpoint/vindrcxr_VICReg_ImageNet_ft \
    --pretrained-how VICReg \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --weight-decay $weightdecay \
    --workers 50 \
    # --resume

# # FT on Shallow RestNet50 RadImagenet
python evaluate_vindrcxr.py \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/vindrcxr_ResNet50_RadImageNet_ft \
    --pretrained-how Supervised \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --weight-decay $weightdecay \
    --workers 50 \
    # --resume

# FT on VICReg RadImageNet
python evaluate_vindrcxr.py \
    --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
    --exp-dir ./checkpoint/vindrcxr_VICReg_RadImageNet_ft \
    --pretrained-how VICReg \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --weight-decay $weightdecay \
    --workers 50 \
    # --resume


# FT on Shallow ResNet50 ImageNet
python evaluate_vindrcxr.py \
    --pretrained_path ./ \
    --exp-dir ./checkpoint/vindrcxr_ResNet50_ImageNet_ft \
    --pretrained-how Supervised \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --weight-decay $weightdecay \
    --workers 50 \
    # --resume
