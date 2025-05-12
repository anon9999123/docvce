from dataclasses import dataclass
from typing import Callable

import imageio
import lpips
import numpy as np
import torch
from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from torch.nn import functional as F
from torchvision.models import vgg19


def _save_tensors(x: torch.Tensor, name: str):
    from torchvision.utils import make_grid

    x = make_grid(x, nrow=int(x.shape[0] ** 0.5), normalize=False)
    imageio.imwrite(
        f"{name}.png", (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    )


def _plot_tensors(x: torch.Tensor, name: str = ""):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    x = make_grid(x, nrow=int(x.shape[0] ** 0.5), normalize=False)
    fig, ax = plt.figure(), plt.gca()
    ax.imshow(x.detach().cpu().permute(1, 2, 0).numpy())
    ax.set_title(name)
    plt.show()


def _renormalize_gradient(grad, eps, small_const=1e-22):
    grad_norm = (
        grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    )
    grad_norm = torch.where(grad_norm < small_const, grad_norm + small_const, grad_norm)
    grad /= grad_norm
    grad *= eps.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    return grad, grad_norm


@dataclass
class GuidedSchedulerConditioningInputs:
    noisy_sample: torch.Tensor = None
    pred_original_sample: torch.Tensor = None
    sqrt_alpha_prod_t: torch.Tensor = None
    sqrt_beta_prod_t: torch.Tensor = None


class Normalizer(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.register_buffer(
            "mu", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "sigma", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

    def forward(self, x):
        x = (torch.clamp(x, -1, 1) + 1) / 2
        x = (x - self.mu) / self.sigma
        return self.classifier(x)


class PerceptualLoss(nn.Module):
    def __init__(self, layer: int = 18):
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        vgg19_model = nn.Sequential(*list(vgg19_model.features.children())[:layer])
        self.model = Normalizer(vgg19_model)
        self.model.eval()

    def forward(self, x0, x1):
        B = x0.size(0)
        l = F.mse_loss(
            self.model(x0).view(B, -1), self.model(x1).view(B, -1), reduction="none"
        ).mean(dim=1)
        return l.sum()


class NoiseGradientGuidance:
    def __init__(
        self,
        device: str = "cpu",
    ):

        super().__init__()
        self.device = device
        self._l1_loss = torch.nn.L1Loss(reduction="sum")
        self._l2_loss = torch.nn.MSELoss(reduction="sum")

    def __call__(
        self,
        original_sample: torch.Tensor,
        noisy_sample: torch.Tensor,
    ):
        with torch.autograd.set_grad_enabled(True):
            loss = 0.0
            loss += self._l1_loss(original_sample, noisy_sample)
            loss += self._l2_loss(original_sample, noisy_sample)
            return torch.autograd.grad(loss, noisy_sample)[0]


class DistanceGradientGuidance:
    def __init__(
        self,
        device: str = "cpu",
        loss_type: str = "vqperceptual",
    ):

        super().__init__()
        self.device = device
        if loss_type == "lpips":
            self.loss_fn = lpips.LPIPS(net="vgg").to(self.device)
        elif loss_type == "vqperceptual":
            self.loss_fn = PerceptualLoss().to(self.device)
        elif loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss(reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def __call__(
        self,
        original_sample: torch.Tensor,
        pred_original_sample: torch.Tensor,
        grad_target: torch.Tensor,
        retain_graph: bool = False,
    ):
        with torch.autograd.set_grad_enabled(True):
            loss = 0.0
            loss += self.loss_fn(
                pred_original_sample,
                original_sample,
            ).sum()
            return torch.autograd.grad(loss, grad_target, retain_graph=retain_graph)[0]


class ClassGradientGuidance:
    def __init__(
        self,
        classifier: Callable,
        target_ind: torch.Tensor,
    ):

        super().__init__()
        self._classifier = classifier
        self._target_ind = target_ind

    def __call__(
        self,
        inputs: torch.Tensor,
        grad_target: torch.Tensor = None,
        retain_graph: bool = False,
    ):
        with torch.autograd.set_grad_enabled(True):
            # runs forward pass
            outputs = self._classifier(inputs)
            selected = outputs[range(outputs.size(0)), self._target_ind.view(-1)]

            assert selected[0].numel() == 1, (
                "Target not provided when necessary, cannot"
                " take gradient with respect to multiple outputs."
            )

            # torch.unbind(forward_out) is a list of scalar tensor tuples and
            # contains batch_size * #steps elements
            if grad_target is not None:
                grads = torch.autograd.grad(
                    selected.sum(), grad_target, retain_graph=retain_graph
                )[0]
            else:
                grads = torch.autograd.grad(
                    selected.sum(), inputs, retain_graph=retain_graph
                )[0]
        # normalized_gradients = normalize(grads)
        # import matplotlib.pyplot as plt

        # plt.imshow(normalized_gradients[0].cpu().permute(1, 2, 0).numpy())
        # plt.title("Normalized Gradients")
        # plt.show()
        return grads, outputs


class SmoothedClassGradientGuidance:
    def __init__(
        self,
        classifier: Callable,
        target_ind: torch.Tensor,
        n_samples: int = 25,
        k: float = 0.15,
        batch_size: int = 10,
    ):
        super().__init__()
        self._classifier = classifier
        self._target_ind = target_ind
        self._n_samples = n_samples
        self._k = k
        self._batch_size = batch_size

    def __call__(
        self,
        inputs: torch.Tensor,
        grad_target: torch.Tensor = None,
        retain_graph: bool = False,
    ):
        stdev = self._k * (inputs.max() - inputs.min())

        grads_accum = (
            torch.zeros_like(grad_target)
            if grad_target is not None
            else torch.zeros_like(inputs)
        )
        outputs_accum = []

        for i in range(0, self._n_samples, self._batch_size):
            current_batch_size = min(self._batch_size, self._n_samples - i)

            with torch.autograd.set_grad_enabled(True):
                repeated_inputs = inputs.repeat_interleave(current_batch_size, dim=0)
                noisy_inputs = (
                    repeated_inputs + torch.randn_like(repeated_inputs) * stdev
                )

                outputs = self._classifier(noisy_inputs)
                outputs_accum.append(outputs)

                selected = outputs[
                    range(outputs.size(0)),
                    self._target_ind.repeat_interleave(current_batch_size, dim=0).view(
                        -1
                    ),
                ]

                if grad_target is not None:
                    grads = torch.autograd.grad(
                        selected.sum(), grad_target, retain_graph=retain_graph
                    )[0]
                else:
                    grads = torch.autograd.grad(
                        selected.sum(), inputs, retain_graph=retain_graph
                    )[0]

            grads_accum += grads / self._n_samples

        outputs_final = torch.cat(outputs_accum, dim=0)

        return grads_accum, outputs_final


# @torch.enable_grad()
# def clean_multiclass_cond_fn(x_t, y, classifier,
#                              s, use_logits, n_samples=25):

#     x_in = x_t.detach().requires_grad_(True)
#     stdev = 0.15 * (x_in.max() - x_in.min())
#     total_gradients = torch.zeros_like(x_in)
#     for i in range(n_samples):
#         noise = torch.randn_like(x_in) * stdev
#         x_plus_noise = x_in + noise
#         x_plus_noise = x_plus_noise.detach().requires_grad_(True)

#         selected = classifier(x_plus_noise)
#         probs = (selected).argmax(-1)

#         # Select the target logits
#         if not use_logits:
#             selected = F.log_softmax(selected, dim=1)
#         selected = -selected[range(len(y)), y]
#         selected = selected * s
#         grads = torch.autograd.grad(selected.sum(), x_plus_noise)[0]
#         total_gradients += grads
#     total_gradients = total_gradients / n_samples
#     return total_gradients, probs


class ClassifierOutputWrapper(torch.nn.Module):
    def __init__(
        self,
        classifier: torch.nn.Module,
        use_logits: bool = False,
    ):
        super().__init__()
        self._classifier = classifier
        self._use_logits = use_logits

    def forward(
        self,
        x: torch.Tensor,
    ):
        logits = self._classifier(x)
        return logits if self._use_logits else F.log_softmax(logits, dim=-1)
