from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import tqdm
from atria.core.utilities.logging import get_logger
from atria.models.task_modules.diffusion.utilities import _unnormalize

from docvce.models.task_modules.diffusion.pipelines.guidance_score_generator import (
    CounterfactualGuidanceScoreGenerator,
)
from docvce.models.task_modules.diffusion.schedulers.guided_ddim import (
    GuidedDDIMScheduler,
)
from docvce.models.task_modules.diffusion.schedulers.guided_ddpm import (
    GuidedDDPMScheduler,
)
from docvce.models.task_modules.diffusion.utilities import (
    ClassGradientGuidance,
    ClassifierOutputWrapper,
    DistanceGradientGuidance,
    NoiseGradientGuidance,
)

logger = get_logger(__name__)


@dataclass
class GuidedCounterfactualDiffusionSamplingPipelineOutput:
    counterfactuals: torch.FloatTensor
    counterfactuals_found: torch.BoolTensor
    counterfactual_logits: torch.FloatTensor
    predicted_counterfactual_labels: torch.LongTensor
    intermediate_samples_at_xt: List[torch.FloatTensor] = None
    intermediate_generated_samples_at_x0: List[torch.FloatTensor] = None
    classifier_outputs: List[torch.FloatTensor] = None


class GuidedCounterfactualDiffusionSamplingPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        classifier: torch.nn.Module,
        scheduler: Union[GuidedDDPMScheduler, GuidedDDIMScheduler],
        vae: torch.nn.Module = None,
        unnormalize_output: bool = True,
        return_intermediate_samples: bool = False,
        total_intermediate_samples: int = 20,
        guidance_scale: float = 1.0,
        use_logits: bool = True,
        # conditioning_function args
        classifier_gradient_weight: float = 1.0,
        distance_gradient_weight: float = 1.0,
        noise_gradient_weight: float = 0.0,
        enable_gradients_renormalization: bool = True,
        use_latent_space_distance: bool = True,
        use_cfg_projection: bool = True,
        cone_projection_type: str = "chunked_zero",
        cone_projection_angle_threshold: float = 45,
        cone_projection_chunk_size: int = 1,
        use_early_stopping: bool = False,
        early_stopping_sampling_interval: int = 10,
    ):
        self._model = model
        self._scheduler = scheduler
        self._classifier = classifier
        self._vae = vae
        self._unnormalize_output = unnormalize_output
        self._return_intermediate_samples = return_intermediate_samples
        self._total_intermediate_samples = total_intermediate_samples
        self._guidance_scale = guidance_scale
        self._use_logits = use_logits
        self._classifier_gradient_weight = classifier_gradient_weight
        self._distance_gradient_weight = distance_gradient_weight
        self._noise_gradient_weight = noise_gradient_weight
        self._enable_gradients_renormalization = enable_gradients_renormalization
        self._use_latent_space_distance = use_latent_space_distance
        self._use_cfg_projection = use_cfg_projection
        self._cone_projection_type = cone_projection_type
        self._cone_projection_angle_threshold = cone_projection_angle_threshold
        self._cone_projection_chunk_size = cone_projection_chunk_size
        self._use_early_stopping = use_early_stopping
        self._early_stopping_sampling_interval = early_stopping_sampling_interval
        self._prepare_models()

    def _prepare_models(self):
        self._classifier = ClassifierOutputWrapper(
            self._classifier,
            use_logits=self._use_logits,
        )

        self._model.requires_grad_(False)
        self._model.eval()

        self._classifier.requires_grad_(False)
        self._classifier.eval()

        if self._vae is not None:
            self._vae.requires_grad_(False)
            self._vae.eval()

    def _prepare_conditioning_score_generator(
        self,
        target_ind: torch.Tensor,
        device: torch.device,
    ):
        return CounterfactualGuidanceScoreGenerator(
            class_gradient_guidance_func=ClassGradientGuidance(
                classifier=self._classifier,
                target_ind=target_ind,
            ),
            distance_gradient_guidance_func=DistanceGradientGuidance(
                device=device,
                loss_type="l2" if self._use_latent_space_distance else "vqperceptual",
            ),
            noise_gradient_guidance_func=(
                NoiseGradientGuidance(
                    device=device,
                )
                if self._vae is None
                else None
            ),
            vae=self._vae,
            scheduler=self._scheduler,
            classifier_gradient_weight=self._classifier_gradient_weight,
            distance_gradient_weight=self._distance_gradient_weight,
            noise_gradient_weight=self._noise_gradient_weight,
            enable_gradients_renormalization=self._enable_gradients_renormalization,
            use_latent_space_distance=self._use_latent_space_distance,
            use_cfg_projection=self._use_cfg_projection,
            cone_projection_type=self._cone_projection_type,
            cone_projection_angle_threshold=self._cone_projection_angle_threshold,
            cone_projection_chunk_size=self._cone_projection_chunk_size,
        )

    def _decode(self, x):
        if self._vae is not None:
            scaling_factor = (
                self._vae.config.scaling_factor
                if hasattr(self._vae, "config")
                else self._vae.scaling_factor
            )
            x = x.cuda()
            x = 1 / scaling_factor * x
            x = self._vae.decode(x).sample
        return x

    def _post_process(self, x):
        x = self._decode(x)
        if self._unnormalize_output:
            x = _unnormalize(x)
        return x

    def _model_forward(self, x: torch.Tensor, t: int, **model_kwargs):
        t = torch.full(
            (x.shape[0],),
            t,
            device=x.device,
            dtype=torch.long,
        )
        model_output = self._model(x, t, **model_kwargs)
        if hasattr(model_output, "sample"):
            model_output = model_output.sample
        return model_output

    def _generate(
        self,
        x: torch.FloatTensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        timesteps: Optional[List[int]] = None,
        **model_kwargs,
    ) -> torch.FloatTensor:
        for t in timesteps:
            # 1. predict noise model_output
            model_output = self._model_forward(x, t, **model_kwargs)

            # 2. compute previous image: x_t -> x_t-1
            x = self._scheduler.step(
                model_output, t, x, generator=generator
            ).prev_sample
        return x

    def _generate_counterfactual(
        self,
        original_sample: torch.Tensor,
        target_query_labels: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **model_kwargs,
    ) -> torch.FloatTensor:
        # x_t is the fully clean image at timestep t
        # z_t is the noisy image at timestep t
        # add noise to x_t -> for t timesteps to get -> z_t
        # at first step, generated_clean_sample coincides with input sample
        samples_at_x0 = original_sample.clone()
        if self._vae is not None:
            from diffusers import AutoencoderKL

            samples_at_x0 = self._vae.encode(samples_at_x0).latent_dist.sample()
            if isinstance(self._vae, AutoencoderKL):
                scaling_factor = self._vae.config.scaling_factor
            else:
                scaling_factor = self._vae.scaling_factor
            samples_at_x0 = samples_at_x0 * scaling_factor
        # _plot_tensors(original_sample, "original_sample")
        # _plot_tensors(samples_at_x0, "samples_at_x0")

        # Sample noise that we'll add to the images
        bsz = samples_at_x0.shape[0]
        noise = torch.randn_like(samples_at_x0)

        # define the starting noise level
        timesteps = (
            torch.ones(
                (bsz,),
                device=samples_at_x0.device,
            ).long()
            * self._scheduler.timesteps[0]
        )
        samples_at_xt = self._scheduler.add_noise(samples_at_x0, noise, timesteps)

        # this is now dime sampling
        logger.info(
            f"Starting guided counterfactual diffusion sampling with timesteps = {self._scheduler.timesteps[:3]} ... {self._scheduler.timesteps[-3:]}"
        )
        logger.info(f"Using logits: {self._classifier._use_logits}")
        logger.info(f"Classifier gradient weight: {self._classifier_gradient_weight}")
        logger.info(f"Noise gradient weight: {self._noise_gradient_weight}")
        logger.info(f"Distance gradient weight: {self._distance_gradient_weight}")
        logger.info(f"Overall Guidance scale: {self._guidance_scale}")

        # setup gradient functions
        conditioning_score_generator = self._prepare_conditioning_score_generator(
            target_ind=target_query_labels,
            device=samples_at_x0.device,
        )

        intermediate_samples_at_xt = []
        intermediate_generated_samples_at_x0 = []
        classifier_outputs = []
        for idx, timestep in tqdm.tqdm(enumerate(self._scheduler.timesteps)):
            if (
                self._use_early_stopping
                and idx > 0
                and idx % self._early_stopping_sampling_interval == 0
            ):  # every n step, we do a full p-sampling run to generated x_0 (generated sample) from z_t (noisy sample)
                early_generated_samples_at_x0 = self._generate(
                    x=samples_at_xt,
                    generator=generator,
                    timesteps=self._scheduler.timesteps[idx + 1 :],
                    **model_kwargs,
                )
                early_generated_samples_at_x0_decoded = self._decode(
                    early_generated_samples_at_x0
                )
                with torch.no_grad():
                    logits = self._classifier(early_generated_samples_at_x0_decoded)
                    predicted_labels = logits.argmax(dim=1)
                    counterfactuals_found = predicted_labels == target_query_labels
                if counterfactuals_found.all():
                    return (
                        early_generated_samples_at_x0.detach(),
                        intermediate_generated_samples_at_x0,
                        intermediate_samples_at_xt,
                        classifier_outputs,
                    )

            samples_at_xt = samples_at_xt.detach().requires_grad_()

            # first we predict the model output
            # this model_output is prediction of either x_t which is x_t_hat, prediction of noise, or prediction of
            # noisy sample at previous step t-1 which is z_{t-1}
            # ultimately we want to predict the z_t-1 which happens inside the guided_step function
            with torch.enable_grad():
                model_output = self._model_forward(
                    samples_at_xt,
                    timestep,
                    **model_kwargs,
                )

            # modl_eoutput_conditional = None
            if conditioning_score_generator.requires_conditional_model_output:
                # cfg projection requires we compute unconditional output and conditional output
                # then same as in guidance_wrapper we'll use the residual of the noise score
                # to guide the diffusion sampling but in this case we use it to project the
                # classifier grads
                model_output_conditional = self._model_forward(
                    samples_at_xt,
                    timestep,
                    **{
                        "class_labels": target_query_labels,
                    },
                )

            # compute the previous noisy sample: z_t -> z_t-1
            step_outputs = self._scheduler.guided_step(
                model_output=model_output,
                timestep=timestep,
                noisy_sample=samples_at_xt,
                guidance_scale=self._guidance_scale,
                conditioning_score_generator=conditioning_score_generator,
                conditioning_score_generator_kwargs=dict(
                    original_sample=samples_at_x0,
                    model_output=model_output,
                    model_output_conditional=(
                        model_output_conditional
                        if conditioning_score_generator.requires_conditional_model_output
                        else None
                    ),
                ),
            )

            # get the generated sample at x_0 and the classifier output
            samples_at_xt, pred_samples_at_x0, classifier_output = (
                step_outputs.prev_sample,
                step_outputs.pred_original_sample,
                step_outputs.classifier_output,
            )
            samples_at_xt = samples_at_xt.detach()
            pred_samples_at_x0 = pred_samples_at_x0.detach()
            if classifier_output is not None:
                classifier_output = classifier_output.detach()

            # if idx % 30 == 0 or idx == len(self._scheduler.timesteps) - 1:
            #     _plot_tensors(
            #         self._post_process(pred_samples_at_x0.detach()),
            #         name="pred_samples_at_x0",
            #     )
            # _plot_tensors(self._post_process(samples_at_xt))

            if self._return_intermediate_samples and (
                idx % 10 == 0 or idx == len(self._scheduler.timesteps) - 1
            ):
                intermediate_generated_samples_at_x0.append(
                    pred_samples_at_x0.detach().cpu()
                )
                intermediate_samples_at_xt.append(samples_at_xt.detach().cpu())
                if classifier_output is not None:
                    classifier_outputs.append(classifier_output.detach().cpu())

        return (
            samples_at_xt,
            intermediate_generated_samples_at_x0,
            intermediate_samples_at_xt,
            classifier_outputs,
        )

    @torch.no_grad()
    def __call__(
        self,
        input: torch.Tensor,
        target_query_labels: torch.Tensor,
        start_timestep: int = 60,
        num_inference_steps: int = 1000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **model_kwargs,
    ) -> GuidedCounterfactualDiffusionSamplingPipelineOutput:
        logger.info(f"Calling the {self.__class__.__name__} with the following args:")
        logger.info(f"start_timestep: {start_timestep}")
        logger.info(f"num_inference_steps: {num_inference_steps}")

        # set step values to be spaced between 0 to num_inference_steps
        self._scheduler.set_timesteps(num_inference_steps)

        # now choose the timesteps to be from start_timestep to num_inference_steps self.scheduler.timesteps right now
        # is from [num_inference_steps, ..., 0], so we take the last start_timestep timesteps
        self._scheduler.timesteps = self._scheduler.timesteps[-start_timestep:]

        # generate from noise
        (
            counterfactuals,
            intermediate_generated_samples_at_x0,
            intermediate_samples_at_xt,
            classifier_outputs,
        ) = self._generate_counterfactual(
            original_sample=input,
            target_query_labels=target_query_labels,
            generator=generator,
            **model_kwargs,
        )

        # counterfactuals
        counterfactuals = self._decode(counterfactuals)

        with torch.no_grad():
            counterfactual_logits = self._classifier(counterfactuals)
            predicted_counterfactual_labels = counterfactual_logits.argmax(dim=1)
            counterfactuals_found = (
                predicted_counterfactual_labels == target_query_labels
            )

        # post process
        if len(intermediate_generated_samples_at_x0) > 0:
            return GuidedCounterfactualDiffusionSamplingPipelineOutput(
                counterfactuals=_unnormalize(counterfactuals),
                counterfactual_logits=counterfactual_logits,
                counterfactuals_found=counterfactuals_found,
                predicted_counterfactual_labels=predicted_counterfactual_labels,
                intermediate_generated_samples_at_x0=[
                    self._post_process(x) for x in intermediate_generated_samples_at_x0
                ],
                intermediate_samples_at_xt=[
                    self._post_process(x) for x in intermediate_samples_at_xt
                ],
                classifier_outputs=classifier_outputs,
            )
        return GuidedCounterfactualDiffusionSamplingPipelineOutput(
            counterfactuals=_unnormalize(counterfactuals),
            counterfactual_logits=counterfactual_logits,
            counterfactuals_found=counterfactuals_found,
            predicted_counterfactual_labels=predicted_counterfactual_labels,
            classifier_outputs=classifier_outputs,
        )
