#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
NAME=sample_latent_diffusion_basic_convnext_base
# ONLY_GENERATE_CONFIGS=${ONLY_GENERATE_CONFIGS:-"True"}
BASE_SCRIPT=$SCRIPT_DIR/../${NAME}.sh

# while [[ "$#" -gt 0 ]]; do
#     case $1 in
#     --only_generate_configs)
#         ONLY_GENERATE_CONFIGS="$2"
#         shift
#         ;;
#     *)
#         echo "Unknown parameter passed: $1"
#         exit 1
#         ;;
#     esac
#     shift
# done

declare -a timesteps=(
    60
    80
    100
)
declare -a classifier_gradient_weight=(
    0.8
    0.7
    0.6
)
declare -a distance_gradient_weight=(
    0.2
    0.3
    0.4
)
declare -a guidance_scale=(
    3.0
    2.0
    1.5
)
declare -a sampling_type=(
    # guided_ddpm
    guided_ddim
)
declare -a cone_projection_type=(
    chunked_zero
    # chunked
    # direct
    # none
)

# generate experiment configs
CONFIGS=()
total_timesteps=${#timesteps[@]}
total_sampling_types=${#sampling_type[@]}
total_cone_projection_type=${#cone_projection_type[@]}
for ((s = 0; s < ${total_sampling_types}; s++)); do
    for ((i = 0; i < ${total_timesteps}; i++)); do
        for ((j = 0; j < ${total_cone_projection_type}; j++)); do
            for ((k = 0; k < ${#guidance_scale[@]}; k++)); do
                for ((l = 0; l < ${#classifier_gradient_weight[@]}; l++)); do
                    if [[ "${cone_projection_type[$j]}" == "none" ]]; then
                        USE_CFG_PROJECTION=False
                    else
                        USE_CFG_PROJECTION=True
                    fi
                    RUN_SCRIPT="$BASE_SCRIPT task_module.start_timestep=${timesteps[$i]} \
                    task_module.classifier_gradient_weight=${classifier_gradient_weight[$l]} \
                    task_module.distance_gradient_weight=${distance_gradient_weight[$l]} \
                    task_module.use_cfg_projection=$USE_CFG_PROJECTION \
                    task_module.cone_projection_type=${cone_projection_type[$j]}\
                    task_module.guidance_scale=${guidance_scale[$k]}\
                    task_module.sampling_type=${sampling_type[$s]}"
                    SCRIPT_NAME=${NAME}_${timesteps[$i]}_${cone_projection_type[$j]}_${guidance_scale[$k]}_${classifier_gradient_weight[$l]}_${distance_gradient_weight[$l]}
                    CONFIGS+=("$SCRIPT_NAME $RUN_SCRIPT")
                done
            done
        done
    done
done

# if [[ $ONLY_GENERATE_CONFIGS == "False" ]]; then
#     # run the configs sequentially
#     total_configs=${#CONFIGS[@]}
#     for ((i = 0; i < ${total_configs}; i++)); do
#         echo "Running script: ${CONFIGS[$i]}"
#         # debug run the first script in configs
#         set -- ${CONFIGS[$i]}
#         EXP_NAME=$1
#         SCRIPT="${@:2}"
#         $SCRIPT
#     done
# fi
