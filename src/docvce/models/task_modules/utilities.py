from dataclasses import dataclass
import torch


@dataclass
class CounterfactualDiffusionModelOutput:
    loss: torch.Tensor = -1
    sample_key: str = None
    real: torch.Tensor = None
    counterfactual: torch.Tensor = None
    real_logits: torch.Tensor = None
    counterfactual_logits: torch.Tensor = None
    real_label: torch.Tensor = None
    counterfactual_label: torch.Tensor = None
