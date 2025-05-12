from atria.core.registry.module_registry import AtriaModuleRegistry

AtriaModuleRegistry.register_metric(
    module="docvce.metrics.cf_eval_metric",
    registered_class_or_func=["CounterfactualEvaluationMetric"],
    name=["cf_eval_metric"],
    device="cpu",
    zen_partial=True,
)
