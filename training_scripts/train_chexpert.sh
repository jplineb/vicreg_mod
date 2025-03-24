#!/bin/sh

#SBATCH --job-name=train_chexpert
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=250gb
#SBATCH --gpus=v100s:1
#SBATCH --partition="work1"

epochs=10
lr_backbone=1e-3
lr_head=1e-2
batch_size=64
weightdecay=1e-6
workers=48
warmup_epochs=2
echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Experiments"
echo "----------------------------"

# Load modules
module load miniforge3/24.3.0-0
module load cuda/12.3

source activate disres
cd /home/jplineb/VICReg/vicreg_mod/

python evaluate_new.py \
    --task_ds chexpert \
    --pretrained_path ./VICReg_ImageNet/resnet50.pth \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how VICReg \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights "freeze" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds chexpert \
    --pretrained_path ./VICReg_ImageNet/resnet50.pth \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how VICReg \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights "finetune"  \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds chexpert \
    --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how VICReg \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights "finetune" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds chexpert \
    --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how VICReg \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights "freeze" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds chexpert \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how Supervised \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights "freeze" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds chexpert \
    --pretrained_path "/project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar"\
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how Supervised \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights "finetune" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds chexpert \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how Supervised \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights "freeze" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds chexpert \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how Supervised \
    --pretrained-dataset ImageNet \
    --epochs $epochs \
    --weights "finetune" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs