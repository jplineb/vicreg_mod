python model_layer_comparison.py \
    --model1-path /home/jplineb/VICReg/vicreg_mod/VICReg_ImageNet/resnet50.pth \
    --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/bcn_20000_VICReg_ImageNet_20250410_173512/checkpoint_20250410_epoch15.pth \
    --layers "0,1,2,3,4,5,6,7" \
    --task-ds bcn_20000 \
    --batch-size 128 \
    --num-samples 10 \
    --visualize