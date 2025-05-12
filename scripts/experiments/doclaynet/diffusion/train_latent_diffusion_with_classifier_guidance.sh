#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

# run training with feature extraction, first run this to extract features but since data loading can be slow
# # the second script can be run after features have been generated
# PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=doclaynet python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
#     hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
#     +trainer_configs=classifier_guidance_latent_diffusion/klf4_with_feature_extraction \
#     +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
#     +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model \
#     task_module.torch_model_builder.reverse_diffusion_model.sample_size=[64,64] \
#     task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
#     task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
#     feature_extraction_batch_size=64 \
#     do_train=False \
#     do_test=False \
#     vae_checkpoint_path=pretrained_models/compvis-kl-f4.ckpt \
#     vae_state_dict_path=state_dict \
#     features_key=compvis-kl-f4 \
#     $@

# # run training, use 1k samples for FID
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=doclaynet python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +trainer_configs=classifier_guidance_latent_diffusion/klf4_preextracted_features \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model \
    task_module.torch_model_builder.reverse_diffusion_model.sample_size=[64,64] \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
    training_engine.max_epochs=200 \
    visualization_engine.visualize_every_n_epochs=10 \
    +data_module.use_train_set_for_test=True \
    data_module.max_test_samples=10000 \
    vae_checkpoint_path=pretrained_models/compvis-kl-f4.ckpt \
    vae_state_dict_path=state_dict \
    features_key=compvis-kl-f4 \
    train_batch_size=32 \
    $@
