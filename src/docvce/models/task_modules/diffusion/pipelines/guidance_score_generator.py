from typing import Callable, Optional

import torch
from atria.core.utilities.logging import get_logger

from docvce.models.task_modules.diffusion.pipelines.cone_projection import (
    cone_project_chunked,
    cone_project_chunked_zero,
    cone_project_direct,
)
from docvce.models.task_modules.diffusion.schedulers.guided_ddim import (
    GuidedDDIMScheduler,
)
from docvce.models.task_modules.diffusion.schedulers.guided_ddpm import (
    GuidedDDPMScheduler,
)
from docvce.models.task_modules.diffusion.utilities import (
    ClassGradientGuidance,
    DistanceGradientGuidance,
    NoiseGradientGuidance,
    _renormalize_gradient,
)

logger = get_logger(__name__)


class CounterfactualGuidanceScoreGenerator:
    def __init__(
        self,
        class_gradient_guidance_func: Optional[ClassGradientGuidance] = None,
        distance_gradient_guidance_func: Optional[DistanceGradientGuidance] = None,
        noise_gradient_guidance_func: Optional[NoiseGradientGuidance] = None,
        vae: Optional[Callable] = None,
        scheduler: Optional[Callable] = None,
        classifier_gradient_weight: float = 1.0,
        distance_gradient_weight: float = 1.0,
        noise_gradient_weight: float = 0.0,
        enable_gradients_renormalization: bool = True,
        use_latent_space_distance: bool = True,
        use_cfg_projection: bool = True,
        cone_projection_type: str = "chunked_zero",
        cone_projection_angle_threshold: float = 45,
        cone_projection_chunk_size: int = 1,
    ):
        self._class_gradient_guidance_func = class_gradient_guidance_func
        self._distance_gradient_guidance_func = distance_gradient_guidance_func
        self._noise_gradient_guidance_func = noise_gradient_guidance_func
        self._vae = vae
        self._scheduler = scheduler
        self._classifier_gradient_weight = classifier_gradient_weight
        self._distance_gradient_weight = distance_gradient_weight
        self._noise_gradient_weight = noise_gradient_weight
        self._use_latent_space_distance = use_latent_space_distance
        self._enable_gradients_renormalization = enable_gradients_renormalization
        self._use_cfg_projection = use_cfg_projection
        self._cone_projection_type = cone_projection_type
        self._cone_projection_angle_threshold = cone_projection_angle_threshold
        self._cone_projection_chunk_size = cone_projection_chunk_size

    @property
    def requires_conditional_model_output(self) -> bool:
        return self._use_cfg_projection

    def _decode(self, x):
        if self._vae is not None:
            scaling_factor = (
                self._vae.config.scaling_factor
                if hasattr(self._vae, "config")
                else self._vae.scaling_factor
            )
            x = x.cuda()
            x = 1 / scaling_factor * x
            x = self._vae.decode(x).sample.clamp(-1, 1)
        return x

    def _project_class_gradients(
        self,
        robust_classifier_gradients: torch.Tensor,
        target_classifier_gradients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Projects the class gradients onto the implicit classifier score using cone projection.

        Args:
            class_gradients (torch.Tensor): Gradients of the loss w.r.t. the classifier.
            model_output (torch.Tensor): Output of the model.
            model_output_conditional (torch.Tensor): Conditional output of the model.
            noisy_sample (torch.Tensor): Noisy sample.

        Returns:
            torch.Tensor: Projected class gradients.
        """
        batch_size = target_classifier_gradients.shape[0]
        with torch.autograd.set_grad_enabled(True):
            assert (
                not robust_classifier_gradients.requires_grad
            ), "implicit_classifier_score requires grad"
            assert (
                not target_classifier_gradients.requires_grad
            ), "classifier_score requires grad"

            # Apply cone projection of class_gradients to the implicit_classifier_score
            if self._cone_projection_type == "chunked_zero":
                projected_class_gradients, consensus = cone_project_chunked_zero(
                    robust_classifier_gradients=robust_classifier_gradients.view(
                        batch_size, -1
                    ),
                    target_classifier_gradients=target_classifier_gradients.view(
                        batch_size, -1
                    ),
                    cone_projection_angle_threshold=self._cone_projection_angle_threshold,
                    base_tensor_shape=target_classifier_gradients.shape,
                    chunk_size=self._cone_projection_chunk_size,
                )
                # projected_class_gradients, _ = _renormalize_gradient(
                #     projected_class_gradients, eps=robust_classifier_gradients
                # )
                # import matplotlib.pyplot as plt

                # # Reshape tensors for plotting
                # projected_class_gradients_flat = (
                #     projected_class_gradients.view(batch_size, -1)
                #     .cpu()
                #     .detach()
                #     .numpy()
                # )
                # robust_classifier_gradients_flat = (
                #     robust_classifier_gradients.view(batch_size, -1)
                #     .cpu()
                #     .detach()
                #     .numpy()
                # )

                # # Ignore zero values
                # non_zero_projected_gradients = projected_class_gradients_flat[
                #     projected_class_gradients_flat != 0
                # ]
                # # Plot histograms
                # plt.figure(figsize=(12, 6))
                # plt.hist(
                #     non_zero_projected_gradients.flatten(),
                #     bins=1000,
                #     alpha=0.5,
                #     label="Projected Class Gradients",
                # )
                # plt.hist(
                #     robust_classifier_gradients_flat.flatten(),
                #     bins=1000,
                #     alpha=0.5,
                #     label="Robust Classifier Gradients",
                # )
                # plt.legend(loc="upper right")
                # plt.title(
                #     "Histogram of Projected Class Gradients vs Robust Classifier Gradients"
                # )
                # plt.xlabel("Gradient Value")
                # plt.ylabel("Frequency")
                # plt.show()

                # _plot_tensors(
                #     consensus.view(
                #         target_classifier_gradients.shape[0],
                #         target_classifier_gradients.shape[2],
                #         target_classifier_gradients.shape[3],
                #     ).float(),
                #     name="consensus",
                # )
                return projected_class_gradients.view_as(target_classifier_gradients)
            elif self._cone_projection_type == "chunked":
                projected_class_gradients, consensus = cone_project_chunked(
                    robust_classifier_gradients=robust_classifier_gradients.view(
                        batch_size, -1
                    ),
                    target_classifier_gradients=target_classifier_gradients.view(
                        batch_size, -1
                    ),
                    cone_projection_angle_threshold=self._cone_projection_angle_threshold,
                    base_tensor_shape=target_classifier_gradients.shape,
                    chunk_size=self._cone_projection_chunk_size,
                )
                # _plot_tensors(
                #     consensus.view(
                #         target_classifier_gradients.shape[0],
                #         target_classifier_gradients.shape[2],
                #         target_classifier_gradients.shape[3],
                #     ).float(),
                #     name="consensus",
                # )
                return projected_class_gradients.view_as(target_classifier_gradients)
            elif self._cone_projection_type == "direct":
                return cone_project_direct(
                    robust_classifier_gradients=robust_classifier_gradients.view(
                        batch_size, -1
                    ),
                    target_classifier_gradients=target_classifier_gradients.view(
                        batch_size, -1
                    ),
                    cone_projection_angle_threshold=self._cone_projection_angle_threshold,
                ).view_as(target_classifier_gradients)
            else:
                raise ValueError(
                    f"Unknown cone projection type: {self._cone_projection_type}"
                )

    def _renormalize_gradients_if_required(
        self,
        gradients: torch.Tensor,
        model_output: torch.Tensor,
        model_output_conditional: torch.Tensor,
    ) -> torch.Tensor:
        if not self._enable_gradients_renormalization:
            return gradients

        if isinstance(self._scheduler, GuidedDDPMScheduler):
            eps = model_output
        elif isinstance(self._scheduler, GuidedDDIMScheduler):
            eps = model_output_conditional  # if not self._use_cfg_projection else cfg_gradients
        else:
            raise NotImplementedError(
                "Only GuidedDDPMScheduler and GuidedDDIMScheduler are supported"
            )
        gradients, _ = _renormalize_gradient(gradients, eps=eps.detach())
        return gradients

    def _compute_class_gradients(
        self,
        pred_original_sample,
        noisy_sample,
        sqrt_beta_prod_t,
        model_output,
        model_output_conditional,
        **kwargs,
    ):
        if self._class_gradient_guidance_func is None:
            return 0.0, None

        # compute classifier gradients against noisy sample
        # _plot_tensors(pred_original_sample, name="pred_original_sample")
        class_gradients, classifier_output = self._class_gradient_guidance_func(
            inputs=pred_original_sample,
            grad_target=noisy_sample,
            retain_graph=True,
        )
        # _plot_tensors(class_gradients / class_gradients.abs().max() * 0.5 + 0.5)

        if isinstance(self._scheduler, GuidedDDIMScheduler):
            # if this is ddim scheduler we need to scale the gradients with sqrt(1-alpha_prod_t)
            class_gradients *= sqrt_beta_prod_t

        if self._use_cfg_projection:
            robust_classifier_gradients = (
                model_output - model_output_conditional
            ).detach()  # this is classifier guidance
            class_gradients = self._project_class_gradients(
                robust_classifier_gradients=robust_classifier_gradients,
                target_classifier_gradients=class_gradients,
            )
        return (
            self._renormalize_gradients_if_required(
                gradients=class_gradients,
                model_output=model_output,
                model_output_conditional=model_output_conditional,
            ),
            classifier_output,
        )

    def _compute_distance_gradients(
        self,
        original_sample,
        pred_original_sample,
        noisy_sample,
        sqrt_beta_prod_t,
        model_output,
        model_output_conditional,
        **kwargs,
    ):
        if self._distance_gradient_guidance_func is None:
            return 0.0

        # _plot_tensors(pred_original_sample, name="pred_original_sample")
        # _plot_tensors(original_sample, name="original_sample")
        distance_gradients = self._distance_gradient_guidance_func(
            original_sample=original_sample,
            pred_original_sample=pred_original_sample,
            grad_target=noisy_sample,
            retain_graph=self._noise_gradient_guidance_func is not None,
        )

        if isinstance(self._scheduler, GuidedDDIMScheduler):
            distance_gradients *= sqrt_beta_prod_t

        return self._renormalize_gradients_if_required(
            gradients=distance_gradients,
            model_output=model_output,
            model_output_conditional=model_output_conditional,
        )

    def _compute_noise_distance_gradients(
        self,
        original_sample,
        noisy_sample,
        sqrt_beta_prod_t,
        model_output,
        model_output_conditional,
        **kwargs,
    ):
        if self._noise_gradient_guidance_func is None:
            return 0.0

        noise_distance_gradients = self._noise_gradient_guidance_func(
            original_sample=original_sample,
            noisy_sample=noisy_sample,
        )

        if isinstance(self._scheduler, GuidedDDIMScheduler):
            noise_distance_gradients *= sqrt_beta_prod_t

        return self._renormalize_gradients_if_required(
            gradients=noise_distance_gradients,
            model_output=model_output,
            model_output_conditional=model_output_conditional,
        )

    def __call__(
        self,
        noisy_sample: torch.Tensor,
        original_sample: torch.Tensor,
        pred_original_sample: torch.Tensor,
        model_output: torch.Tensor = None,
        model_output_conditional: torch.Tensor = None,
        sqrt_beta_prod_t: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        decoded_original_sample = None
        decoded_pred_original_sample = None
        if self._vae is not None:
            with torch.enable_grad():
                if not self._use_latent_space_distance:
                    decoded_original_sample = self._decode(original_sample)
                decoded_pred_original_sample = self._decode(pred_original_sample)

        # Compute the gradients for classifier-guidance
        class_gradients, classifier_output = self._compute_class_gradients(
            pred_original_sample=(
                decoded_pred_original_sample
                if self._vae is not None
                else pred_original_sample
            ),
            noisy_sample=noisy_sample,
            sqrt_beta_prod_t=sqrt_beta_prod_t,
            model_output=model_output,
            model_output_conditional=model_output_conditional,
            **kwargs,
        )
        # _plot_tensors(class_gradients / class_gradients.abs().max() * 0.5 + 0.5)

        # Compute the gradients for distance-guidance
        distance_gradients = self._compute_distance_gradients(
            original_sample=(
                decoded_original_sample
                if self._vae is not None and not self._use_latent_space_distance
                else original_sample
            ),
            pred_original_sample=(
                decoded_pred_original_sample
                if self._vae is not None and not self._use_latent_space_distance
                else pred_original_sample
            ),
            noisy_sample=noisy_sample,
            sqrt_beta_prod_t=sqrt_beta_prod_t,
            model_output=model_output,
            model_output_conditional=model_output_conditional,
            **kwargs,
        )
        # _plot_tensors(distance_gradients / distance_gradients.abs().max() * 0.5 + 0.5)

        # Compute the gradients for noise-gradient guidance
        noise_distance_gradients = self._compute_noise_distance_gradients(
            original_sample=original_sample,
            noisy_sample=noisy_sample,
            sqrt_beta_prod_t=sqrt_beta_prod_t,
            model_output=model_output,
            model_output_conditional=model_output_conditional,
            **kwargs,
        )

        # Combine the gradients
        gradients = (
            class_gradients * self._classifier_gradient_weight
            - self._distance_gradient_weight * distance_gradients
            - self._noise_gradient_weight * noise_distance_gradients
        )
        return gradients, classifier_output
