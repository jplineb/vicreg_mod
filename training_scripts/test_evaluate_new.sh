epochs=10
lr_backbone=0.001
lr_head=0.03
batch_size=64
weights='finetune'
weightdecay=1e-6

echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Linear Probe Experiments"
echo "----------------------------"

# Load modules
module load miniforge3/24.3.0-0
module load cuda/12.3

source activate disres
cd /home/jplineb/VICReg/vicreg_mod/

python evaluate_new.py \
    --task_ds messidor \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how Supervised \
    --pretrained-dataset ImageNet \
    --epochs 10 \
    --weights "freeze" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 12

python evaluate_new.py \
    --task_ds messidor \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how Supervised \
    --pretrained-dataset ImageNet \
    --epochs 10 \
    --weights "finetune"  \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 12

python evaluate_new.py \
    --task_ds messidor \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how Supervised \
    --pretrained-dataset RadImageNet \
    --epochs 10 \
    --weights "finetune" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 12 

python evaluate_new.py \
    --task_ds messidor \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/try_new_script \
    --pretrained-how Supervised \
    --pretrained-dataset RadImageNet \
    --epochs 10 \
    --weights "freeze" \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 12 