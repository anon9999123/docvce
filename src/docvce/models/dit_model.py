import torch
from torch import nn
from transformers import BeitModel


class DitModel(nn.Module):
    def __init__(self, num_labels: int = 16):
        super().__init__()
        self.model = BeitModel.from_pretrained(
            "microsoft/dit-base", add_pooling_layer=True
        )
        self.classifier = (
            nn.Linear(self.model.config.hidden_size, num_labels)
            if num_labels > 0
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        return self.classifier(outputs.pooler_output)
