#!/bin/sh

# SBATCH --job-name=train_messidor
# SBATCH --ntasks=1
# SBATCH --time=06:00:00
# SBATCH --cpus-per-task=32
# SBATCH --mem=125gb
# SBATCH --gpus=v100s:1

epochs=20
lr_backbone=1e-3
lr_head=1e-3
batch_size=64
weights='finetune'
weightdecay=1e-6
workers=16

echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Linear Probe Experiments"
echo "----------------------------"

# Load modules
module load miniforge3/24.3.0-0
module load cuda/12.3

source activate disres
cd /home/jplineb/VICReg/vicreg_mod/

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs $epochs \
#     --weights "freeze" \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers $workers

python evaluate_new.py \
    --task_ds bcn_20000 \
    --pretrained_path ./VICReg_ImageNet/resnet50.pth \
    --pretrained-how VICReg \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights "finetune"  \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers

python evaluate_new.py \
    --task_ds bcn_20000 \
    --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
    --pretrained-how VICReg \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights "finetune" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers 

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset RadImageNet \
#     --epochs $epochs \
#     --weights "freeze" \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers $workers 


# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
#     --pretrained-how Supervised \
#     --pretrained-dataset RadImageNet \
#     --epochs $epochs \
#     --weights "freeze" \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers $workers

python evaluate_new.py \
    --task_ds bcn_20000 \
    --pretrained_path "/project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar"\
    --pretrained-how Supervised \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights "finetune" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers 

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how Supervised \
#     --pretrained-dataset ImageNet \
#     --epochs $epochs \
#     --weights "freeze" \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers $workers 

python evaluate_new.py \
    --task_ds bcn_20000 \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --pretrained-how Supervised \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights "finetune" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers 


########################################################

### LR TUNNING ###
########################################################

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs 20 \
#     --weights "finetune"  \
#     --lr-backbone 5e-6 \
#     --lr-head 1e-4\
#     --batch-size $batch_size \
#     --workers 12

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs 20 \
#     --weights "finetune"  \
#     --lr-backbone 1e-5 \
#     --lr-head 1e-4\
#     --batch-size $batch_size \
#     --workers 12

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs 20 \
#     --weights "finetune"  \
#     --lr-backbone 1e-4 \
#     --lr-head 1e-4\
#     --batch-size $batch_size \
#     --workers 12

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs 20 \
#     --weights "finetune"  \
#     --lr-backbone 1e-3 \
#     --lr-head 1e-4\
#     --batch-size $batch_size \
#     --workers 12

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs 20 \
#     --weights "finetune"  \
#     --lr-backbone 1e-3 \
#     --lr-head 1e-3 \
#     --batch-size $batch_size \
#     --workers 12

# python evaluate_new.py \
#     --task_ds bcn_20000 \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs 20 \
#     --weights "freeze"  \
#     --lr-backbone 1e-3 \
#     --lr-head 1e-3 \
#     --batch-size $batch_size \
#     --workers 12