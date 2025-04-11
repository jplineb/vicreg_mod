# python model_layer_comparison.py \
#     --model1-path /home/jplineb/VICReg/vicreg_mod/VICReg_ImageNet/resnet50.pth \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/bcn_20000_VICReg_ImageNet_20250410_173512/checkpoint_20250410_epoch15.pth \
#     --layers "0,1,2,3,4,5,6,7" \
#     --task-ds bcn_20000 \
#     --batch-size 128 \
#     --num-samples 10 \
#     --output-dir ./layer_comparisons/bcn2k_VICReg_ImageNet \
#     --visualize

# python model_layer_comparison.py \
#     --model1-path /home/jplineb/VICReg/vicreg_mod/VICReg_ImageNet/resnet50.pth \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/messidor_VICReg_ImageNet_20250410_180141/checkpoint_20250410_epoch13.pth \
#     --layers "0,1,2,3,4,5,6,7" \
#     --task-ds messidor \
#     --batch-size 128 \
#     --num-samples 10 \
#     --output-dir ./layer_comparisons/messidor_VICReg_ImageNet \
#     --visualize

# python model_layer_comparison.py \
#     --model1-path /home/jplineb/VICReg/vicreg_mod/checkpoint/messidor_VICReg_RadImageNet_20250410_180733/checkpoint_20250410_epoch30.pth\
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/messidor_VICReg_ImageNet_20250410_180141/checkpoint_20250410_epoch13.pth \
#     --layers "0,1,2,3,4,5,6,7" \
#     --task-ds messidor \
#     --batch-size 128 \
#     --num-samples 10 \
#     --output-dir ./layer_comparisons/messidor_VICReg_ImageNet_VS_RadImageNet \
#     --visualize

python model_layer_comparison.py \
    --model1-path /home/jplineb/VICReg/vicreg_mod/checkpoint/bcn_20000_VICReg_RadImageNet_20250410_174656/checkpoint_20250410_epoch20.pth \
    --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/bcn_20000_VICReg_ImageNet_20250410_173512/checkpoint_20250410_epoch15.pth \
    --layers "0,1,2,3,4,5,6,7" \
    --task-ds bcn_20000 \
    --batch-size 128 \
    --num-samples 10 \
    --output-dir ./layer_comparisons/bcn2k_VICReg_ImageNet_VS_RadImageNet \
    --visualize