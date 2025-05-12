#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

# python src/docvce/compute_metrics_for_hpr.py --experiment_dir ./output/counterfactual_latent_diffusion/ds4sd_DocLayNet/basic_convnext_base-refined3 --model_type convnext_base --model_path ./pretrained_models/doclaynet/convnext_base.pt --dataset doclaynet
# python src/docvce/compute_metrics_for_hpr.py --experiment_dir ./output/counterfactual_latent_diffusion/ds4sd_DocLayNet/basic_resnet50-refined3 --model_type resnet50 --model_path ./pretrained_models/doclaynet/resnet50.pt --dataset doclaynet
python src/docvce/compute_metrics_for_hpr.py --experiment_dir ./output/counterfactual_latent_diffusion/ds4sd_DocLayNet/basic_docvce.models.dit_model.DitModel-refined3 --model_type dit_b --model_path ./pretrained_models/doclaynet/dit_b.pt --dataset doclaynet
