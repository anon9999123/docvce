#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

# # run eval
MODEL=convnext_base
CLASSIFIER_MODEL_CHECKPOINT=pretrained_models/convnext_base_rvlcdip_basic.pt
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_model_evaluator.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
    +trainer_configs=image_classification/evaluate \
    torch_model_builder@task_module.torch_model_builder=timm \
    +task_module.torch_model_builder.model_name=convnext_base \
    gray_to_rgb=True \
    image_size=256 \
    classifier_checkpoint_path=$CLASSIFIER_MODEL_CHECKPOINT \
    classifier_state_dict_path=task_module.model \
    +train_batch_size=64 \
    +eval_batch_size=64 \
    $ARGS

# # run eval
# CLASSIFIER_MODEL_CHECKPOINT=pretrained_models/convnext_base_rvlcdip_robust.pt
# PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_model_evaluator.py \
#     hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://docvce/conf] \
#     +trainer_configs=image_classification/evaluate \
#     gray_to_rgb=True \
#     image_size=256 \
#     torch_model_builder@task_module.torch_model_builder=local \
#     task_module.torch_model_builder.model_name=docvce.models.task_modules.convnext.convnext_base \
#     classifier_checkpoint_path=$CLASSIFIER_MODEL_CHECKPOINT \
#     classifier_state_dict_path=ema_state_dict.model \
#     +train_batch_size=64 \
#     +eval_batch_size=64 \
#     $ARGS
