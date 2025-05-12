#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH
DIFFUSION_MODEL_CHECKPOINT=output/atria_trainer/RvlCdip/classifier_guidance_klf4/checkpoints/checkpoint_399720.pt
MODEL=(
    resnet50
    # convnext_base
    docvce.models.dit_model.DitModel
)
MODEL_TYPE=(
    timm
    # timm
    local
)
CLASSIFIER_MODEL_CHECKPOINT=(
    pretrained_models/rvlcdip/resnet50.pt
    # pretrained_models/rvlcdip/convnext_base.pt
    pretrained_models/rvlcdip/dit_b.pt
)
samples=(
    "guided_ddpm"
    "guided_ddim"
)
if [ ! -f $DIFFUSION_MODEL_CHECKPOINT ]; then
    echo "Diffusion model checkpoint not found: $DIFFUSION_MODEL_CHECKPOINT"
    exit 1
fi
for checkpoint in "${CLASSIFIER_MODEL_CHECKPOINT[@]}"; do
    if [ ! -f $checkpoint ]; then
        echo "Classifier model checkpoint not found: $checkpoint"
        exit 1
    fi
done
for i in "${!MODEL[@]}"; do
    model=${MODEL[$i]}
    model_type=${MODEL_TYPE[$i]}
    checkpoint=${CLASSIFIER_MODEL_CHECKPOINT[$i]}
    for sampling_type in "${samples[@]}"; do
        echo $model $model_type $checkpoint
        PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip python ./external/atria/src/atria/core/task_runners/atria_model_evaluator.py \
            hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
            +cf_configs=counterfactual_latent_diffusion/guided_klf4 \
            +task_module.torch_model_builder.reverse_diffusion_model.model_name=UNet2DModel \
            +model_config@task_module.torch_model_builder.reverse_diffusion_model=unet_2d_model \
            task_module.torch_model_builder.reverse_diffusion_model.sample_size=[64,64] \
            task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
            task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
            torch_model_builder@task_module.torch_model_builder.classifier=$model_type \
            task_module.torch_model_builder.classifier.model_name=$model \
            diffusion_model_checkpoint_path=$DIFFUSION_MODEL_CHECKPOINT \
            classifier_checkpoint_path=$checkpoint \
            classifier_state_dict_path=task_module.model \
            classifier_output_name=basic_$model \
            task_module.sampling_type=$sampling_type \
            eval_batch_size=16 \
            task_module.start_timestep=60 \
            task_module.guidance_scale=3.0 \
            task_module.classifier_gradient_weight=0.7 \
            task_module.distance_gradient_weight=0.3 \
            data_module.max_test_samples=10000 \
            task_module.target_query_labels_path=$BASE_PATH/notebooks/class_similarity/rvlcdip/top_5_closest_classes.csv
        $@
    done
done
