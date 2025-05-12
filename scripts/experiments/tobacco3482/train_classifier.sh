#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

# run training
MODEL=convnext_base
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482 python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +trainer_configs=image_classification/train \
    torch_model_builder@task_module.torch_model_builder=timm \
    +task_module.torch_model_builder.model_name=$MODEL \
    gray_to_rgb=True \
    image_size=224 \
    +train_batch_size=64 \
    +eval_batch_size=64 \
    experiment_name=no_pretrained

# run training
MODEL=resnet50
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482 python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +trainer_configs=image_classification/train \
    torch_model_builder@task_module.torch_model_builder=timm \
    +task_module.torch_model_builder.model_name=$MODEL \
    gray_to_rgb=True \
    image_size=224 \
    +train_batch_size=64 \
    +eval_batch_size=64 \
    experiment_name=no_pretrained

# run training
MODEL=docvce.models.dit_model.DitModel
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482 python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +trainer_configs=image_classification/train \
    torch_model_builder@task_module.torch_model_builder=local \
    +task_module.torch_model_builder.model_name=$MODEL \
    gray_to_rgb=True \
    image_size=224 \
    +train_batch_size=64 \
    +eval_batch_size=64 \
    experiment_name=no_pretrained
