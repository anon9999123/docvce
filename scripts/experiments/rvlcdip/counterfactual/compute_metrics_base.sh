#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

$SCRIPT_DIR/generate_sfid_split.sh real_samples_dir_path=./output/counterfactual_latent_diffusion/RvlCdip/basic_convnext_base/
$SCRIPT_DIR/generate_sfid_split.sh real_samples_dir_path=./output/counterfactual_latent_diffusion/RvlCdip/basic_resnet50/
$SCRIPT_DIR/generate_sfid_split.sh real_samples_dir_path=./output/counterfactual_latent_diffusion/RvlCdip/basic_docvce.models.dit_model.DitModel/

PYTHONPATH=$PYTHONPATH python $BASE_PATH/src/docvce/compute_metrics.py --experiment_dir ./output/counterfactual_latent_diffusion/RvlCdip/basic_convnext_base
PYTHONPATH=$PYTHONPATH python $BASE_PATH/src/docvce/compute_metrics.py --experiment_dir ./output/counterfactual_latent_diffusion/RvlCdip/basic_resnet50
PYTHONPATH=$PYTHONPATH python $BASE_PATH/src/docvce/compute_metrics.py --experiment_dir ./output/counterfactual_latent_diffusion/RvlCdip/basic_docvce.models.dit_model.DitModel/
