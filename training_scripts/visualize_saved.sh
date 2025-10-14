#!/usr/bin/env bash
set -euo pipefail

# Usage examples (adjust names as desired)
# Messidor experiments
python render_visualizations.py \
  --input-dir ./layer_comparisons/messidor_Base_Supervised_ImageNet_VS_Supervised_Imagenet \
  --model1-name "Base S ImageNet" \
  --model2-name "messidor S ImageNet" \
  --use-features || true

python render_visualizations.py \
  --input-dir ./layer_comparisons/messidor_Base_VICREG_ImageNet_VS_VICREG_ImageNet \
  --model1-name "Base SSL ImageNet" \
  --model2-name "messidor SSL ImageNet" \
  --use-features || true

python render_visualizations.py \
  --input-dir ./layer_comparisons/messidor_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet \
  --model1-name "Base S RadImageNet" \
  --model2-name "messidor S RadImageNet" \
  --use-features || true

python render_visualizations.py \
  --input-dir ./layer_comparisons/messidor_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet \
  --model1-name "Base SSL RadImageNet" \
  --model2-name "messidor SSL RadImageNet" \
  --use-features || true 