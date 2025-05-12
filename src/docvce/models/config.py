from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register diffusers task modules type=[CounterfactualDiffusionModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__)
    + ".task_modules.diffusion.counterfactual_diffusion",
    name="counterfactual_diffusion",
    registered_class_or_func="CounterfactualDiffusionModule",
    hydra_defaults=[
        "_self_",
        {
            "/torch_model_builder@torch_model_builder.reverse_diffusion_model": "diffusers",
        },
        {
            "/torch_model_builder@torch_model_builder.classifier": "local",
        },
    ],
)

# register diffusers task modules type=[CounterfactualDiffusionModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__)
    + ".task_modules.diffusion.counterfactual_latent_diffusion",
    name="counterfactual_latent_diffusion",
    registered_class_or_func="CounterfactualLatentDiffusionModule",
    hydra_defaults=[
        "_self_",
        {
            "/torch_model_builder@torch_model_builder.reverse_diffusion_model": "diffusers",
        },
        {"/torch_model_builder@torch_model_builder.vae": "diffusers"},
        {
            "/torch_model_builder@torch_model_builder.classifier": "local",
        },
    ],
)

# register latent classification task modules type=[LatentClassificationModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__)
    + ".task_modules.latent_classification_module.latent_classification_module",
    name="latent_classification",
    registered_class_or_func="LatentClassificationModule",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder.vae": "diffusers"},
        {
            "/torch_model_builder@torch_model_builder.classifier": "local",
        },
    ],
)
