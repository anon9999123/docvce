#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

# run training with feature extraction, first run this to extract features but since data loading can be slow
# the second script can be run after features have been generated
# DIFFUSION_MODEL_CHECKPOINT=output/atria_trainer/RvlCdip/classifier_guidance_klf4/checkpoints/checkpoint_399720.pt
# PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482 python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
#     hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
#     +trainer_configs=classifier_guidance_latent_diffusion/klf4_with_feature_extraction \
#     +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
#     +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model \
#     task_module.torch_model_builder.reverse_diffusion_model.sample_size=[64,64] \
#     task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
#     task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
#     pretrained_checkpoint_path=$DIFFUSION_MODEL_CHECKPOINT \
#     do_train=False \
#     do_test=False \
#     $@

# run training, use 1k samples for FID
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482 python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +trainer_configs=classifier_guidance_latent_diffusion/klf4_preextracted_features \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model \
    task_module.torch_model_builder.reverse_diffusion_model.sample_size=[64,64] \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
    pretrained_checkpoint_path=$DIFFUSION_MODEL_CHECKPOINT \
    training_engine.max_epochs=500 \
    visualization_engine.visualize_every_n_epochs=100 \
    +data_module.use_train_set_for_test=True \
    data_module.max_test_samples=1000 \
    $@
