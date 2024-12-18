epochs=5
lr_backbone=0.001
lr_head=0.03
batch_size=20
weights='finetune'

echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Linear Probe Experiments"
echo "----------------------------"

# Load modules
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0

source activate pda
cd /home/jplineb/VICReg/vicreg_mod/

# Linear Probe on Shallow RestNet50 RadImagenet
# python evaluate_vindrcxr.py \
#     --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
#     --exp-dir ./checkpoint/chexpert_ResNet50_RadImageNet_TEST_RUN \
#     --pretrained-how Supervised \
#     --pretrained-dataset RadImageNet \
#     --epochs $epochs \
#     --weights $weights \
#     --lr-backbone $lr_backbone \
#     --lr-head $lr_head \
#     --batch-size $batch_size \
#     --workers 8 \
#     # --resume

python evaluate_chexpert.py \
    --pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar \
    --exp-dir ./checkpoint/test \
    --pretrained-how Supervised \
    --pretrained-dataset RadImageNet \
    --epochs $epochs \
    --weights $weights \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers 8 \
    # --resume