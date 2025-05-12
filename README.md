# DocVCE: Diffusion-based Visual Counterfactual Explanations for Document Image Classification
This is a source code for the paper submission to ICDAR 2025.

# Getting started
Install the project dependencies.
```
pip install -r ./external/atria/requirements.txt
pip install -r ./external/docsets/requirements.txt
```

# Train the classifier
Prepare the specified dataset [doclaynet, tobacco3482, rvlcdip] and train the classifier on it. This script will start to train all 3 models in sequence.
```
./scripts/experiments/<dataset>/train_classifier.sh
```

# Train the diffusion models to generate document images based on latent-diffusion
Prepare the specified dataset [doclaynet, tobacco3482, rvlcdip] and train a latent-diffusion model on it. This script will train the diffusion model on specified datasets. 
```
./scripts/experiments/tobacco3482/diffusion/train_latent_diffusion_with_classifier_guidance.sh
```

Generate new samples if needed:
```
sample_latent_diffusion_with_classifier_guidance
```

For full training, dataset, and model configurations see the following files
1. Training: src/docvce/conf/trainer_configs/classifier_guidance_latent_diffusion.yaml
2. Model: src/docvce/conf/model_config/unet_2d_model.yaml
3. Dataset: src/docvce/conf/data_module/document_classification/*

# Generate base counterfactual examples
Generate counterfactual examples for the test set. 
```
./scripts/experiments/<dataset>/counterfactual/sample_latent_diffusion.sh
```
For full set of configuration parameters used for counterfactual generation, see the configs.
```
src/docvce/conf/cf_configs/counterfactual_latent_diffusion/guided_klf4.yaml
```

# Generate refined counterfactual examples
Generate refined counterfactual examples for the test set. 
```
./scripts/experiments/<dataset>/counterfactual/compute_refined_counterfactuals.sh
```
See both full naive implementation and efficient batch implementation of the HPR module in the following script.
```
./src/docvce/hierarchical_patch_wise_refinement_batch.py
```

# Generate sFID split for evaluation 
Using this script you can generate sFID split:
```
./scripts/experiments/tobacco3482/counterfactual/generate_sfid_split.sh
```

# Evaluation of all metrics on the generated counterfactuals
You can use the following scripts to generate all metric results evaluation
```
./scripts/experiments/<dataset>/counterfactual/compute_metrics_base.sh
```

# Evaluation of all metrics on the refined counterfactuals
You can use the following scripts to generate all metric results evaluation
```
./scripts/experiments/<dataset>/counterfactual/compute_metrics_refined.sh
```

# Class similarity
The following folder contains class similarity computed based on SimSiam that can be used for counterfactual generation.
```
notebooks/class_similarity/<dataset>
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.