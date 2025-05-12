#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

# run sampling
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +trainer_configs=classifier_guidance_latent_diffusion/klf4_online_feature_extraction \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model \
    task_module.torch_model_builder.reverse_diffusion_model.sample_size=[64,64] \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
    task_module.guidance_scale=5.0 \
    eval_batch_size=4 \
    do_train=False \
    $@
