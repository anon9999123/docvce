CONFIGS=()
total_target_run_index=55
for ((n = 0; n < ${total_target_run_index}; n++)); do
    CONFIGS+=("python src/docvce/hierarchical_patch_wise_refinement_batch.py --experiment_dir ./output/counterfactual_latent_diffusion/RvlCdip/basic_convnext_base/ --model_checkpoint ./pretrained_models/convnext_base_rvlcdip_basic.pt --target_run_index $n")
done

# total_configs=${#CONFIGS[@]}
# for ((i = 0; i < ${total_configs}; i++)); do
#     echo "Running script: ${CONFIGS[$i]}"
#     eval ${CONFIGS[$i]}
# done
