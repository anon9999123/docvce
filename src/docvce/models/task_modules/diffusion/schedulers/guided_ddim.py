from dataclasses import dataclass
from typing import Callable, Tuple, Union

import torch
from dacite import Optional
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class GuidedDDIMSchedulerOutput:
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None
    classifier_output: torch.Tensor = None


class GuidedDDIMScheduler(DDIMScheduler):
    def guided_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        noisy_sample: torch.Tensor,
        conditioning_score_generator: Optional[Callable] = None,
        conditioning_score_generator_kwargs: Optional[dict] = None,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[GuidedDDIMSchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        with torch.enable_grad():
            if self.config.prediction_type == "epsilon":
                pred_original_sample = (
                    noisy_sample - beta_prod_t ** (0.5) * model_output
                ) / alpha_prod_t ** (0.5)
                pred_epsilon = model_output
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
                pred_epsilon = (
                    noisy_sample - alpha_prod_t ** (0.5) * pred_original_sample
                ) / beta_prod_t ** (0.5)
            elif self.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_sample - (
                    beta_prod_t**0.5
                ) * model_output
                pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                    beta_prod_t**0.5
                ) * noisy_sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction`"
                )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (
                noisy_sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)

        # 6. Guide the sample based on a classifier or a distance gradient function
        if conditioning_score_generator is not None:
            gradients, classifier_output = conditioning_score_generator(
                noisy_sample=noisy_sample,
                pred_original_sample=pred_original_sample,
                sqrt_alpha_prod_t=alpha_prod_t ** (0.5),
                sqrt_beta_prod_t=beta_prod_t ** (0.5),
                **conditioning_score_generator_kwargs,
            )

            # guide pred_epsilon
            # guidance is done as eps - eps * sqrt_beta_prod_t * gradients but we send this into condition_fn
            # so it applies the scaling there before gradient renormalization
            pred_epsilon = pred_epsilon - guidance_scale * gradients

            # now predict new original sample
            pred_original_sample = (
                noisy_sample - beta_prod_t ** (0.5) * pred_epsilon
            ) / alpha_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return GuidedDDIMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
            classifier_output=classifier_output,
        )
