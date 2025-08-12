# Chexpert
# Check before and after
# python model_layer_comparison.py \
#     --model1-name "Base S ImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/chexpert_Supervised_ImageNet_20250411_114359/checkpoint_20250411_epoch4.pth \
#     --model2-name "Chexpert S ImageNet" \
#     --task-ds chexpert \
#     --batch-size 25 \
#     --num-samples 50 \
#     --output-dir ./layer_comparisons/chexpert_Base_Supervised_ImageNet_VS_Supervised_Imagenet \

# python model_layer_comparison.py \
#     --model1-path /home/jplineb/VICReg/vicreg_mod/VICReg_ImageNet/resnet50.pth \
#     --model1-name "Base SSL ImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/chexpert_VICReg_ImageNet_20250411_092451/checkpoint_20250411_epoch4.pth \
#     --model2-name "Chexpert SSL ImageNet" \
#     --task-ds chexpert \
#     --batch-size 25 \
#     --num-samples 50 \
#     --output-dir ./layer_comparisons/chexpert_Base_VICREG_ImageNet_VS_VICREG_ImageNet \

# python model_layer_comparison.py \
#     --model1-path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar\
#     --model1-name "Base S RadImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/chexpert_Supervised_RadImageNet_20250411_114401/checkpoint_20250411_epoch3.pth  \
#     --model2-name "Chexpert S RadImageNet" \
#     --task-ds chexpert \
#     --batch-size 25 \
#     --num-samples 50 \
#     --output-dir ./layer_comparisons/chexpert_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet \

# python model_layer_comparison.py \
#     --model1-path ./VICReg_RadImageNet/resnet50.pth \
#     --model1-name "Base SSL RadImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/chexpert_VICReg_RadImageNet_20250411_092742/checkpoint_20250411_epoch5.pth  \
#     --model2-name "Chexpert SSL RadImageNet" \
#     --task-ds chexpert \
#     --batch-size 25 \
#     --num-samples 50 \
#     --output-dir ./layer_comparisons/chexpert_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet \






# # VINDR
# python model_layer_comparison.py \
#     --model1-name "Base S ImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/vindrcxr_Supervised_ImageNet_20250411_125610/checkpoint_20250411_epoch4.pth \
#     --model2-name "vindrcxr S ImageNet" \
#     --task-ds vindrcxr \
#     --batch-size 25 \
#     --num-samples 50 \
#     --output-dir ./layer_comparisons/vindrcxr_Base_Supervised_ImageNet_VS_Supervised_Imagenet \

# python model_layer_comparison.py \
#     --model1-path /home/jplineb/VICReg/vicreg_mod/VICReg_ImageNet/resnet50.pth \
#     --model1-name "Base SSL ImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/vindrcxr_VICReg_ImageNet_20250411_115212/checkpoint_20250411_epoch5.pth \
#     --model2-name "vindrcxr SSL ImageNet" \
#     --task-ds vindrcxr \
#     --batch-size 25 \
#     --num-samples 50 \
#     --output-dir ./layer_comparisons/vindrcxr_Base_VICREG_ImageNet_VS_VICREG_ImageNet \

# python model_layer_comparison.py \
#     --model1-path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar\
#     --model1-name "Base S RadImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/vindrcxr_Supervised_RadImageNet_20250411_123518/checkpoint_20250411_epoch5.pth  \
#     --model2-name "vindrcxr S RadImageNet" \
#     --task-ds vindrcxr \
#     --batch-size 25 \
#     --num-samples 50 \
#     --output-dir ./layer_comparisons/vindrcxr_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet \

# python model_layer_comparison.py \
#     --model1-path ./VICReg_RadImageNet/resnet50.pth \
#     --model1-name "Base SSL RadImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/vindrcxr_VICReg_RadImageNet_20250411_120529/checkpoint_20250411_epoch2.pth \
#     --model2-name "vindrcxr SSL RadImageNet" \
#     --task-ds vindrcxr \
#     --batch-size 25 \
#     --num-samples 50 \
#     --output-dir ./layer_comparisons/vindrcxr_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet \




# bcn2k
# python model_layer_comparison.py \
#     --model1-name "Base S ImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/bcn_20000_Supervised_ImageNet_20250715_090613/best_epoch_20250715.pth \
#     --model2-name "bcn2k S ImageNet" \
#     --task-ds bcn_20000 \
#     --batch-size 25 \
#     --num-samples 100 \
#     --output-dir ./layer_comparisons/bcn2k_Base_Supervised_ImageNet_VS_Supervised_Imagenet \

# python model_layer_comparison.py \
#     --model1-path /home/jplineb/VICReg/vicreg_mod/VICReg_ImageNet/resnet50.pth \
#     --model1-name "Base SSL ImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/bcn_20000_VICReg_ImageNet_20250715_075226/best_epoch_20250715.pth \
#     --model2-name "bcn2k SSL ImageNet" \
#     --task-ds bcn_20000 \
#     --batch-size 25 \
#     --num-samples 100 \
#     --output-dir ./layer_comparisons/bcn2k_Base_VICREG_ImageNet_VS_VICREG_ImageNet \

# python model_layer_comparison.py \
#     --model1-path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar\
#     --model1-name "Base S RadImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/bcn_20000_Supervised_RadImageNet_20250715_084139/best_epoch_20250715.pth \
#     --model2-name "bcn2k S RadImageNet" \
#     --task-ds bcn_20000 \
#     --batch-size 25 \
#     --num-samples 100 \
#     --output-dir ./layer_comparisons/bcn2k_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet \

# python model_layer_comparison.py \
#     --model1-path ./VICReg_RadImageNet/resnet50.pth \
#     --model1-name "Base SSL RadImageNet" \
#     --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/bcn_20000_VICReg_RadImageNet_20250715_081706/best_epoch_20250715.pth\
#     --model2-name "bcn2k SSL RadImageNet" \
#     --task-ds bcn_20000 \
#     --batch-size 25 \
#     --num-samples 100 \
#     --output-dir ./layer_comparisons/bcn2k_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet \

# messidor
python model_layer_comparison.py \
    --model1-name "Base S ImageNet" \
    --model2-path  /home/jplineb/VICReg/vicreg_mod/checkpoint/messidor_Supervised_RadImageNet_20250801_084141/best_epoch_20250801.pth \
    --model2-name "messidor S ImageNet" \
    --task-ds messidor \
    --batch-size 25 \
    --num-samples 100 \
    --output-dir ./layer_comparisons/messidor_Base_Supervised_ImageNet_VS_Supervised_Imagenet \

python model_layer_comparison.py \
    --model1-path /home/jplineb/VICReg/vicreg_mod/VICReg_ImageNet/resnet50.pth \
    --model1-name "Base SSL ImageNet" \
    --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/messidor_VICReg_ImageNet_20250801_083350/best_epoch_20250801.pth \
    --model2-name "messidor SSL ImageNet" \
    --task-ds messidor \
    --batch-size 25 \
    --num-samples 100 \
    --output-dir ./layer_comparisons/messidor_Base_VICREG_ImageNet_VS_VICREG_ImageNet \

python model_layer_comparison.py \
    --model1-path /project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar\
    --model1-name "Base S RadImageNet" \
    --model2-path  \
    --model2-name "messidor S RadImageNet" \
    --task-ds messidor \
    --batch-size 25 \
    --num-samples 100 \
    --output-dir ./layer_comparisons/messidor_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet \

python model_layer_comparison.py \
    --model1-path ./VICReg_RadImageNet/resnet50.pth \
    --model1-name "Base SSL RadImageNet" \
    --model2-path /home/jplineb/VICReg/vicreg_mod/checkpoint/messidor_VICReg_RadImageNet_20250801_083742/best_epoch_20250801.pth \
    --model2-name "messidor SSL RadImageNet" \
    --task-ds messidor \
    --batch-size 25 \
    --num-samples 100 \
    --output-dir ./layer_comparisons/messidor_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet \
