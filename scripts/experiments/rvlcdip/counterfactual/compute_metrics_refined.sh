#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

python src/docvce/compute_metrics_for_hpr.py --experiment_dir ./output/counterfactual_latent_diffusion/RvlCdip/basic_convnext_base-refined3 --model_type convnext_base --model_path ./pretrained_models/rvlcdip/convnext_base.pt --dataset rvlcdip
python src/docvce/compute_metrics_for_hpr.py --experiment_dir ./output/counterfactual_latent_diffusion/RvlCdip/basic_resnet50-refined3 --model_type resnet50 --model_path ./pretrained_models/rvlcdip/resnet50.pt --dataset rvlcdip
python src/docvce/compute_metrics_for_hpr.py --experiment_dir ./output/counterfactual_latent_diffusion/RvlCdip/basic_docvce.models.dit_model.DitModel-refined3 --model_type dit_b --model_path ./pretrained_models/rvlcdip/dit_b.pt --dataset rvlcdip
