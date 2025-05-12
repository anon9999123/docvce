#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH
DIFFUSION_MODEL_CHECKPOINT=pretrained_models/unconditional_ddpm_rvlcdip.pt
CLASSIFIER_MODEL_CHECKPOINT=pretrained_models/convnext_base_rvlcdip.pt
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip python ./external/atria/src/atria/core/task_runners/atria_model_evaluator.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +cf_configs=counterfactual_diffusion/guided \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model_v2 \
    task_module.torch_model_builder.reverse_diffusion_model.sample_size=[256,256] \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
    torch_model_builder@task_module.torch_model_builder.classifier=timm \
    task_module.torch_model_builder.classifier.model_name=convnext_base \
    diffusion_model_checkpoint_path=$DIFFUSION_MODEL_CHECKPOINT \
    classifier_checkpoint_path=$CLASSIFIER_MODEL_CHECKPOINT \
    image_size=256 \
    sampling_type=guided_ddim \
    $@
