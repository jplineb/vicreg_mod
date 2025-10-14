#!/bin/sh

# SBATCH --job-name=train_messidor
# SBATCH --ntasks=1
# SBATCH --time=06:00:00
# SBATCH --cpus-per-task=32
# SBATCH --mem=125gb
# SBATCH --gpus=a100:1
# SBATCH --partition=wficai

epochs=15
lr_backbone=1e-3
lr_head=1e-2
batch_size=64
weightdecay=1e-6
workers=16
warmup_epochs=5
echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Experiments"
echo "----------------------------"

# Load modules
module load miniforge3/24.3.0-0
module load cuda/12.3
module list

source activate disres
cd /home/jplineb/VICReg/vicreg_mod/

# python evaluate_new.py \
#     --task_ds messidor \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs $epochs \
#     --weights "freeze" \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers $workers \
#     --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds messidor \
    --pretrained_path ./VICReg_ImageNet/resnet50.pth \
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
    --task_ds messidor \
    --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
    --pretrained-how VICReg \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights "finetune" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs

# python evaluate_new.py \
#     --task_ds messidor \
#     --pretrained_path ./VICReg_RadImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset RadImageNet \
#     --epochs $epochs \
#     --weights "freeze" \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers $workers \
#     --warmup-epochs $warmup_epochs

# python evaluate_new.py \
#     --task_ds messidor \
#     --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how Supervised \
#     --pretrained-dataset RadImageNet \
#     --epochs $epochs \
#     --weights "freeze" \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers $workers \
#     --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds messidor \
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

# python evaluate_new.py \
#     --task_ds messidor \
#     --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how Supervised \
#     --pretrained-dataset ImageNet \
#     --epochs $epochs \
#     --weights "freeze" \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers $workers \
#     --warmup-epochs $warmup_epochs

python evaluate_new.py \
    --task_ds messidor \
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


########################################################

### LR TUNNING ###
########################################################

# python evaluate_new.py \
#     --task_ds messidor \
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
#     --task_ds messidor \
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
#     --task_ds messidor \
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
#     --task_ds messidor \
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
#     --task_ds messidor \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs 20 \
#     --weights "finetune"  \
#     --lr-backbone 1e-3 \
#     --lr-head 1e-3 \
#     --batch-size $batch_size \
#     --workers 16

# python evaluate_new.py \
#     --task_ds messidor \
#     --pretrained_path ./VICReg_ImageNet/resnet50.pth \
#     --exp-dir ./checkpoint/try_new_script \
#     --pretrained-how VICReg \
#     --pretrained-dataset ImageNet \
#     --epochs 20 \
#     --weights "freeze"  \
#     --lr-backbone 1e-3 \
#     --lr-head 1e-3 \
#     --batch-size $batch_size \
#     --workers 16