#!/bin/sh

#SBATCH --job-name=train_chexpert
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=250gb
#SBATCH --gpus=v100s:1
#SBATCH --partition="work1"

epochs=15
lr_backbone=1e-4
lr_head=1e-3
batch_size=128
weightdecay=1e-6
workers=48
warmup_epochs=3
echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Performing Experiments"
echo "----------------------------"

# Load modules
module load miniforge3/24.3.0-0
module load cuda/12.3

source activate disres
cd /home/jplineb/VICReg/vicreg_mod/

# Define configurations
configs=(
    # "--pretrained_path ./VICReg_ImageNet/resnet50.pth --pretrained-how VICReg --pretrained-dataset ImageNet --weights freeze"
    "--pretrained_path ./VICReg_ImageNet/resnet50.pth --pretrained-how VICReg --pretrained-dataset ImageNet --weights finetune"
    "--pretrained_path ./VICReg_RadImageNet/resnet50.pth --pretrained-how VICReg --pretrained-dataset RadImageNet --weights finetune"
    # "--pretrained_path ./VICReg_RadImageNet/resnet50.pth --pretrained-how VICReg --pretrained-dataset RadImageNet --weights freeze"
    # "--pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar --pretrained-how Supervised --pretrained-dataset RadImageNet --weights freeze"
    "--pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar --pretrained-how Supervised --pretrained-dataset RadImageNet --weights finetune"
    # "--pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar --pretrained-how Supervised --pretrained-dataset ImageNet --weights freeze"
    "--pretrained_path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar --pretrained-how Supervised --pretrained-dataset ImageNet --weights finetune"
)

# Submit a job for each configuration
for config in "${configs[@]}"; do
    sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=train_chexpert
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=250gb
#SBATCH --gpus=v100s:1
#SBATCH --partition="work1"

module load miniforge3/24.3.0-0
module load cuda/12.3

source activate disres
cd /home/jplineb/VICReg/vicreg_mod/

python evaluate_new.py \
    --task_ds chexpert \
    --epochs $epochs \
    --lr-backbone $lr_backbone \
    --lr-head $lr_head \
    --batch-size $batch_size \
    --workers $workers \
    --warmup-epochs $warmup_epochs \
    $config
EOT
done