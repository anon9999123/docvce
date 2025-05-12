from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register diffusers task modules type=[CounterfactualDiffusionModule]
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".doclaynet_label_transform",
    name="doclaynet_label_transform",
    registered_class_or_func="DoclayNetLabelTransform",
)

# register diffusers task modules type=[CounterfactualDiffusionModule]
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".hocr_extract",
    name="hocr_extract_transform",
    registered_class_or_func="HocrExtractTransform",
)
