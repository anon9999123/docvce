#!/bin/bash
python src/docvce/hierarchical_patch_wise_refinement_batch.py \
    --experiment_dir ./output/counterfactual_latent_diffusion/Tobacco3482/basic_convnext_base/ \
    --model_checkpoint ./pretrained_models/tobacco3482/convnext_base.pt \
    --model_type convnext_base \
    --num_classes 10 \
    --target_run_index 0-2
python src/docvce/hierarchical_patch_wise_refinement_batch.py \
    --experiment_dir ./output/counterfactual_latent_diffusion/Tobacco3482/basic_resnet50/ \
    --model_checkpoint ./pretrained_models/tobacco3482/resnet50.pt \
    --model_type resnet50 \
    --num_classes 10 \
    --target_run_index 0-2
python src/docvce/hierarchical_patch_wise_refinement_batch.py \
    --experiment_dir ./output/counterfactual_latent_diffusion/Tobacco3482/basic_docvce.models.dit_model.DitModel/ \
    --model_checkpoint ./pretrained_models/tobacco3482/dit_b.pt \
    --model_type dit_b \
    --num_classes 10 \
    --target_run_index 0-2
