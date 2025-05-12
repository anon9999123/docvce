#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH
DIFFUSION_MODEL_CHECKPOINT=output/atria_trainer/RvlCdip/classifier_guidance_klf4/checkpoints/checkpoint_399720.pt
CLASSIFIER_MODEL_CHECKPOINT=pretrained_models/convnext_base_rvlcdip_basic.pt
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip python ./external/atria/src/atria/core/task_runners/atria_model_evaluator.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +cf_configs=counterfactual_latent_diffusion/guided_klf4 \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model \
    task_module.torch_model_builder.reverse_diffusion_model.sample_size=[64,64] \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
    torch_model_builder@task_module.torch_model_builder.classifier=timm \
    task_module.torch_model_builder.classifier.model_name=convnext_base \
    diffusion_model_checkpoint_path=$DIFFUSION_MODEL_CHECKPOINT \
    classifier_checkpoint_path=$CLASSIFIER_MODEL_CHECKPOINT \
    classifier_state_dict_path=task_module.model \
    classifier_output_name=basic_convnext_base \
    task_module.sampling_type=guided_ddim \
    $@
