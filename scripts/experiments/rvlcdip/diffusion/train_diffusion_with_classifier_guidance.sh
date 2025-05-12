#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

# run training
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +trainer_configs=classifier_guidance_diffusion/train \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model_v2 \
    task_module.torch_model_builder.reverse_diffusion_model.sample_size=[256,256] \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
    experiment_name=classifier_guidance_diffusion_v2 \
    image_size=256 \
    $@
