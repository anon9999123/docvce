from dataclasses import dataclass
from typing import Callable, Tuple, Union

import torch
from dacite import Optional
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class GuidedDDPMSchedulerOutput:
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None
    classifier_output: torch.Tensor = None


class GuidedDDPMScheduler(DDPMScheduler):
    def guided_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        noisy_sample: torch.Tensor,
        conditioning_score_generator: Optional[Callable] = None,
        conditioning_score_generator_kwargs: Optional[dict] = None,
        guidance_scale: float = 1.0,
        generator=None,
        return_dict: bool = True,
    ) -> Union[GuidedDDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == noisy_sample.shape[
            1
        ] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(
                model_output, noisy_sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        with torch.enable_grad():
            if self.config.prediction_type == "epsilon":
                pred_original_sample = (
                    noisy_sample - beta_prod_t ** (0.5) * model_output
                ) / alpha_prod_t ** (0.5)
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_sample - (
                    beta_prod_t**0.5
                ) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                    " `v_prediction`  for the DDPMScheduler."
                )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * noisy_sample
        )

        # 6. Guide the sample based on a classifier or a distance gradient function
        if conditioning_score_generator is not None:
            gradients, classifier_output = conditioning_score_generator(
                noisy_sample=noisy_sample,
                pred_original_sample=pred_original_sample,
                sqrt_alpha_prod_t=alpha_prod_t ** (0.5),
                sqrt_beta_prod_t=beta_prod_t ** (0.5),
                **conditioning_score_generator_kwargs,
            )

            # Add the gradients to the predicted previous sample for guidance
            # shift the mean by the gradients * variance as done in 'guided-diffusion'
            # new_mean = mean + variance * grad
            # _plot_tensors(grads.cpu() / grads.cpu().abs().max() * 0.5 + 0.5)
            pred_prev_sample = (
                pred_prev_sample
                + guidance_scale
                * self._get_variance(t, predicted_variance=predicted_variance)
                * gradients
            )

        # 7. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            if self.variance_type == "fixed_small_log":
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance)
                    * variance_noise
                )
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (
                    self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
                ) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
                classifier_output,
            )

        return GuidedDDPMSchedulerOutput(
            prev_sample=pred_prev_sample,
            pred_original_sample=pred_original_sample,
            classifier_output=classifier_output,
        )

    # def guided_step_with_recurrence(
    #     self,
    #     model_output: torch.Tensor,
    #     timestep: int,
    #     noisy_sample: torch.Tensor,
    #     original_sample: torch.Tensor,
    #     class_gradient_guidance_func: Optional[Callable] = None,
    #     distance_gradient_guidance_func: Optional[Callable] = None,
    #     noise_gradient_guidance_func: Optional[Callable] = None,
    #     recurrence_steps: int = 10,
    #     generator=None,
    #     return_dict: bool = True,
    # ) -> Union[GuidedDDPMEstGradsSchedulerOutput, Tuple]:
    #     """
    #     Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    #     process from the learned model outputs (most often the predicted noise).

    #     Args:
    #         model_output (`torch.Tensor`):
    #             The direct output from learned diffusion model.
    #         timestep (`float`):
    #             The current discrete timestep in the diffusion chain.
    #         sample (`torch.Tensor`):
    #             A current instance of a sample created by the diffusion process.
    #         generator (`torch.Generator`, *optional*):
    #             A random number generator.
    #         return_dict (`bool`, *optional*, defaults to `True`):
    #             Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

    #     Returns:
    #         [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
    #             If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
    #             tuple is returned where the first element is the sample tensor.

    #     """
    #     t = timestep

    #     prev_t = self.previous_timestep(t)

    #     if model_output.shape[1] == noisy_sample.shape[
    #         1
    #     ] * 2 and self.variance_type in [
    #         "learned",
    #         "learned_range",
    #     ]:
    #         model_output, predicted_variance = torch.split(
    #             model_output, noisy_sample.shape[1], dim=1
    #         )
    #     else:
    #         predicted_variance = None

    #     # 1. compute alphas, betas
    #     alpha_prod_t = self.alphas_cumprod[t]
    #     alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
    #     beta_prod_t = 1 - alpha_prod_t
    #     beta_prod_t_prev = 1 - alpha_prod_t_prev
    #     current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    #     current_beta_t = 1 - current_alpha_t

    #     for recurrence_step in tqdm.tqdm(range(recurrence_steps), "Recurrence steps"):
    #         # 2. compute predicted original sample from predicted noise also called
    #         # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    #         if self.config.prediction_type == "epsilon":
    #             pred_original_sample = (
    #                 noisy_sample - beta_prod_t ** (0.5) * model_output
    #             ) / alpha_prod_t ** (0.5)
    #         elif self.config.prediction_type == "sample":
    #             pred_original_sample = model_output
    #         elif self.config.prediction_type == "v_prediction":
    #             pred_original_sample = (alpha_prod_t**0.5) * noisy_sample - (
    #                 beta_prod_t**0.5
    #             ) * model_output
    #         else:
    #             raise ValueError(
    #                 f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
    #                 " `v_prediction`  for the DDPMScheduler."
    #             )

    #         # 3. Clip or threshold "predicted x_0"
    #         if self.config.thresholding:
    #             pred_original_sample = self._threshold_sample(pred_original_sample)
    #         elif self.config.clip_sample:
    #             pred_original_sample = pred_original_sample.clamp(
    #                 -self.config.clip_sample_range, self.config.clip_sample_range
    #             )

    #         # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    #         # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    #         pred_original_sample_coeff = (
    #             alpha_prod_t_prev ** (0.5) * current_beta_t
    #         ) / beta_prod_t
    #         current_sample_coeff = (
    #             current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
    #         )

    #         # 5. Compute predicted previous sample µ_t
    #         # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    #         pred_prev_sample = (
    #             pred_original_sample_coeff * pred_original_sample
    #             + current_sample_coeff * noisy_sample
    #         )

    #         # 6. Guide the sample based on a classifier or a distance gradient function
    #         grads, classifier_output = self._compute_guidance_gradients(
    #             noisy_sample=noisy_sample,
    #             original_sample=original_sample,
    #             pred_original_sample=pred_original_sample,
    #             class_gradient_guidance_func=class_gradient_guidance_func,
    #             distance_gradient_guidance_func=distance_gradient_guidance_func,
    #             noise_gradient_guidance_func=noise_gradient_guidance_func,
    #             sqrt_alpha_prod_t=alpha_prod_t ** (0.5),
    #         )

    #         # Add the gradients to the predicted previous sample for guidance
    #         # shift the mean by the gradients * variance as done in 'guided-diffusion'
    #         # new_mean = mean + variance * grad
    #         pred_prev_sample = (
    #             pred_prev_sample
    #             + self._get_variance(t, predicted_variance=predicted_variance) * grads
    #         )

    #         # 7. Add noise
    #         variance = 0
    #         if t > 0:
    #             device = model_output.device
    #             variance_noise = randn_tensor(
    #                 model_output.shape,
    #                 generator=generator,
    #                 device=device,
    #                 dtype=model_output.dtype,
    #             )
    #             if self.variance_type == "fixed_small_log":
    #                 variance = (
    #                     self._get_variance(t, predicted_variance=predicted_variance)
    #                     * variance_noise
    #                 )
    #             elif self.variance_type == "learned_range":
    #                 variance = self._get_variance(
    #                     t, predicted_variance=predicted_variance
    #                 )
    #                 variance = torch.exp(0.5 * variance) * variance_noise
    #             else:
    #                 variance = (
    #                     self._get_variance(t, predicted_variance=predicted_variance)
    #                     ** 0.5
    #                 ) * variance_noise

    #         pred_prev_sample = pred_prev_sample + variance

    #         if recurrence_step < recurrence_steps - 1:
    #             # 6. again add noise to predicted sample to generate noisy sample and repeat...
    #             noise = randn_tensor(
    #                 noisy_sample.shape,
    #                 generator=generator,
    #                 device=noisy_sample.device,
    #                 dtype=noisy_sample.dtype,
    #             )
    #             noisy_sample = (
    #                 self.alphas[t] ** (0.5) * pred_prev_sample
    #                 + (1 - self.alphas[t]) ** (0.5) * noise
    #             )

    #     if not return_dict:
    #         return (
    #             pred_prev_sample,
    #             pred_original_sample,
    #             classifier_output,
    #         )

    #     return GuidedDDPMEstGradsSchedulerOutput(
    #         prev_sample=pred_prev_sample,
    #         pred_original_sample=pred_original_sample,
    #         classifier_output=classifier_output,
    #     )
