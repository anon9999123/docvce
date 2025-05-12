# pytorch_diffusion + derived encoder decoder
import math
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ignite.distributed as idist
import pandas as pd
import torch
import torchvision
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ModelOutput
from atria.core.models.task_modules.atria_task_module import (
    CheckpointConfig,
    TorchModelDict,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.diffusion.latent_diffusion import LatentDiffusionModule
from atria.models.task_modules.diffusion.utilities import _unnormalize
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine

from docvce.models.task_modules.diffusion.pipelines.guided_counter_factual_diffusion_sampling import (
    GuidedCounterfactualDiffusionSamplingPipeline,
    GuidedCounterfactualDiffusionSamplingPipelineOutput,
)
from docvce.models.task_modules.diffusion.schedulers.guided_ddim import (
    GuidedDDIMScheduler,
)
from docvce.models.task_modules.diffusion.schedulers.guided_ddpm import (
    GuidedDDPMScheduler,
)
from docvce.models.task_modules.utilities import CounterfactualDiffusionModelOutput

logger = get_logger(__name__)


class CounterfactualLatentDiffusionModule(LatentDiffusionModule):
    _REQUIRES_BUILDER_DICT = True
    _SUPPORTED_BUILDERS = [
        "LocalTorchModelBuilder",
        "DiffusersModelBuilder",
        "TimmModelBuilder",
    ]

    def __init__(
        self,
        torch_model_builder: Union[
            partial[TorchModelBuilderBase], Dict[str, partial[TorchModelBuilderBase]]
        ],
        checkpoint_configs: Optional[List[CheckpointConfig]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        # diffusion args
        diffusion_steps: int = 1000,
        inference_diffusion_steps: int = 200,
        # class conditioning args
        enable_class_conditioning: bool = False,
        use_cfg: bool = False,
        # ldm args
        latent_input_key: str = DataKeys.LATENT_IMAGE,
        use_precomputed_latents_if_available: bool = False,
        features_scale_factor: Optional[float] = None,
        # counterfactual diffusion args
        sampling_type: str = "guided_ddpm",
        target_query_labels: Optional[Union[int, List[int]]] = None,
        target_query_labels_path: Optional[Union[str, Path]] = None,
        start_timestep: int = 60,
        guidance_scale: float = 1.0,
        use_logits: bool = False,
        # conditioning_function args
        classifier_gradient_weight: float = 0.6,
        distance_gradient_weight: float = 0.2,
        noise_gradient_weight: float = 0.0,
        enable_gradients_renormalization: bool = True,
        use_latent_space_distance: bool = True,
        use_cfg_projection: bool = True,
        cone_projection_type: str = "chunked_zero",
        cone_projection_angle_threshold: float = 45,
        cone_projection_chunk_size: int = 1,
        # logging args
        enable_tensorboard_logging: bool = False,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
            diffusion_steps=diffusion_steps,
            inference_diffusion_steps=inference_diffusion_steps,
            enable_class_conditioning=enable_class_conditioning,
            use_cfg=use_cfg,
            clip_sample=False,
            latent_input_key=latent_input_key,
            use_precomputed_latents_if_available=use_precomputed_latents_if_available,
            features_scale_factor=features_scale_factor,
        )
        self._sampling_type = sampling_type
        self._target_query_labels = target_query_labels
        self._target_query_labels_path = target_query_labels_path
        self._start_timestep = start_timestep
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
        self._enable_tensorboard_logging = enable_tensorboard_logging

        # for this model, both enable_class_conditioning and use_cfg must be used together or not
        if enable_class_conditioning:
            assert use_cfg, "CFG must be enabled when class conditioning is enabled"
        if use_cfg_projection:
            assert (
                enable_class_conditioning and use_cfg
            ), "CFG projection must be enabled when class conditioning and CFG are enabled"

    def _build_model(self) -> Union[torch.nn.Module, TorchModelDict]:
        model: TorchModelDict = super()._build_model()
        assert hasattr(
            model.trainable_models, "classifier"
        ), "The trainable_models models must contain a `classifier` underlying model."
        model.trainable_models.reverse_diffusion_model.requires_grad_(False)
        model.trainable_models.reverse_diffusion_model.eval()
        model.trainable_models.classifier.requires_grad_(False)
        model.trainable_models.classifier.eval()
        model.non_trainable_models.vae.requires_grad_(False)
        model.non_trainable_models.vae.eval()
        return model

    def _setup_model_config(self) -> Dict[str, Dict[str, Any]]:
        if not isinstance(self._torch_model_builder, dict):
            raise NotImplementedError(
                self.__class__.__name__ + " requires a dict of TorchModelBuilders"
            )
        assert "classifier" in self._torch_model_builder, (
            "The torch model builder must be a dict containing `classifier` underlying model. "
            "You must define it through the config file."
        )
        assert (
            self._dataset_metadata.labels is not None
        ), "The dataset metadata must contain labels. "
        classifier_kwargs = dict(
            num_labels=len(self._dataset_metadata.labels),
        )
        reverse_diffusion_model_kwargs = {}
        if self._enable_class_conditioning:
            num_class_embeds = len(self._dataset_metadata.labels)
            num_class_embeds = (
                num_class_embeds + 1 if self._use_cfg else num_class_embeds
            )
            reverse_diffusion_model_kwargs = dict(
                num_class_embeds=num_class_embeds,
                num_labels=num_class_embeds,
                num_classes=num_class_embeds,
            )

        kwargs = {}
        for key in self._torch_model_builder:
            if key == "classifier":
                kwargs["classifier"] = classifier_kwargs
            elif key == "reverse_diffusion_model":
                kwargs["reverse_diffusion_model"] = reverse_diffusion_model_kwargs
            else:
                kwargs[key] = {}
        return kwargs

    def build_forward_noise_scheduler(self):
        if self._sampling_type == "guided_ddpm":
            scheduler_class = GuidedDDPMScheduler
        elif self._sampling_type == "guided_ddim":
            scheduler_class = GuidedDDIMScheduler
        else:
            raise ValueError(
                f"Sampling type {self._sampling_type} not supported. "
                f"Supported sampling types are ['guided_ddpm', 'guided_ddim']"
            )

        # initialize the scheduler
        forward_noise_scheduler = scheduler_class(
            num_train_timesteps=self._diffusion_steps,
            beta_schedule=self._noise_schedule,
            prediction_type=self._objective,
            clip_sample=self._clip_sample,
            clip_sample_range=self._clip_sample_range,
        )

        # set the number of timesteps for inference
        forward_noise_scheduler.set_timesteps(self._inference_diffusion_steps)

        # Get the target for loss depending on the prediction type
        if self._objective is not None:
            # set objective of scheduler if defined
            forward_noise_scheduler.register_to_config(prediction_type=self._objective)

        logger.info(
            f"Loaded {scheduler_class} forward noise scheduler with parameters:"
        )
        logger.info(f"Number of diffusion steps: {self._diffusion_steps}")
        logger.info(f"Objective: {self._objective}")
        logger.info(f"Schedule type: {self._noise_schedule}")
        logger.info(f"Clip sample: {self._clip_sample}")
        logger.info(f"Clip sample range: {self._clip_sample_range}")
        return forward_noise_scheduler

    def _build_sampling_pipeline(
        self,
        model: torch.nn.Module,
        return_intermediate_samples: bool = False,
    ) -> GuidedCounterfactualDiffusionSamplingPipeline:
        if self._sampling_type in [
            "guided_ddpm",
            "guided_ddim",
        ]:
            return GuidedCounterfactualDiffusionSamplingPipeline(
                model=model.trainable_models.reverse_diffusion_model,
                classifier=model.trainable_models.classifier,
                vae=model.non_trainable_models.vae,
                scheduler=self._forward_noise_scheduler,
                unnormalize_output=self._unnormalize_output,
                return_intermediate_samples=return_intermediate_samples
                and self._enable_tensorboard_logging,
                use_logits=self._use_logits,
                guidance_scale=self._guidance_scale,
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
        else:
            raise ValueError(
                f"Sampling type {self._sampling_type} not supported. "
                f"Supported sampling types are ['guided_ddpm', 'guided_ddim']"
            )

    def _prepare_model_kwargs(
        self, batch: BatchDict, stage: TrainingStage
    ) -> torch.Tensor:
        if self._enable_class_conditioning:
            # when class condtioning is enabled, we just ask model to generate samples unconditionally using the null
            # class label which is = num_labels
            class_labels = batch[DataKeys.LABEL]
            no_class_label = len(self._dataset_metadata.labels) * torch.ones_like(
                class_labels, device=class_labels.device
            )
            return {
                "class_labels": no_class_label,
            }
        else:
            return {}

    def _generate_data(
        self,
        input: torch.Tensor,
        target_query_labels: torch.Tensor,
        return_intermediate_samples: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        sampling_pipeline = self._build_sampling_pipeline(
            model=self._torch_model,
            return_intermediate_samples=return_intermediate_samples,
        )
        return sampling_pipeline(
            input=input,
            target_query_labels=target_query_labels,
            start_timestep=self._start_timestep,
            num_inference_steps=self._inference_diffusion_steps,
            **model_kwargs,
        )

    def _prepare_target_query_labels(
        self, input: torch.Tensor, ground_truth_labels: torch.Tensor
    ) -> torch.Tensor:
        if self._target_query_labels is None and self._target_query_labels_path is None:
            return ground_truth_labels

        if self._target_query_labels_path is not None:
            target_query_labels_per_class = pd.read_csv(self._target_query_labels_path)
            target_query_label_batch = []
            for gt_label in ground_truth_labels:
                possible_target_classes = torch.tensor(
                    target_query_labels_per_class[
                        target_query_labels_per_class["Class"] == gt_label.item()
                    ].values[0][1:]
                )
                rand_indices = torch.randperm(possible_target_classes.size(0))
                target_query_label = possible_target_classes[rand_indices][0]
                target_query_label_batch.append(target_query_label)
            return torch.stack(target_query_label_batch).to(input.device)

        if isinstance(self._target_query_labels, int):
            return torch.tensor([self._target_query_labels] * input.shape[0]).to(
                input.device
            )
        elif isinstance(self._target_query_labels, list):
            assert (
                len(self._target_query_labels) == input.shape[0]
            ), f"Target query labels length mismatch. Expected {input.shape[0]}, "
            return torch.tensor(self._target_query_labels).to(input.device)

    def _log_data_to_tensorboard(
        self,
        input,
        pipeline_outputs: GuidedCounterfactualDiffusionSamplingPipelineOutput,
        iteration: int,
        ground_truth_labels: torch.Tensor,
        target_query_labels: torch.Tensor,
    ):
        generated_samples = pipeline_outputs.generated_samples
        intermediate_generated_samples_at_x0 = (
            pipeline_outputs.intermediate_generated_samples_at_x0
        )
        intermediate_samples_at_xt = pipeline_outputs.intermediate_samples_at_xt
        if idist.get_rank() == 0:
            classifier_outputs = pipeline_outputs.classifier_outputs
            for step_idx, classifier_output in enumerate(classifier_outputs):
                ground_truth_outputs = torch.gather(
                    classifier_output, -1, ground_truth_labels.unsqueeze(-1).cpu()
                )
                target_query_outputs = torch.gather(
                    classifier_output, -1, target_query_labels.unsqueeze(-1).cpu()
                )
                for idx, (ground_truth_output, target_query_output) in enumerate(
                    zip(ground_truth_outputs, target_query_outputs)
                ):
                    self._tb_logger.writer.add_scalar(
                        f"ground_truth_outputs/iter_{iteration}/sample_{idx}/gt_{ground_truth_labels[idx]}",
                        ground_truth_output.item(),
                        step_idx,
                    )
                    self._tb_logger.writer.add_scalar(
                        f"target_query_outputs/iter_{iteration}/sample_{idx}/target_{target_query_labels[idx]}",
                        target_query_output.item(),
                        step_idx,
                    )

            logger.info("Adding image batch to tensorboard")
            self._tb_logger.writer.add_image(
                f"visualization/{self._input_key}",
                torchvision.utils.make_grid(
                    input,
                    normalize=False,
                    nrow=int(math.sqrt(generated_samples.shape[0])),
                ),
                iteration,
            )

            # add intermediate clean samples to tensorboard
            for idx, intermediate_sample in enumerate(
                intermediate_generated_samples_at_x0
            ):
                self._tb_logger.writer.add_image(
                    f"visualization/intermediate_generated_samples_at_x0_{iteration}/{self._classifier_gradient_weight}",
                    torchvision.utils.make_grid(
                        intermediate_sample,
                        normalize=False,
                        nrow=int(math.sqrt(intermediate_sample.shape[0])),
                    ),
                    idx,
                )

            # add intermediate samples to tensorboard
            for idx, intermediate_sample in enumerate(intermediate_samples_at_xt):
                print(f"Adding image ={idx}", intermediate_sample.shape)
                self._tb_logger.writer.add_image(
                    f"visualization/intermediate_samples_at_xt_{iteration}/{self._classifier_gradient_weight}",
                    torchvision.utils.make_grid(
                        intermediate_sample,
                        normalize=False,
                        nrow=int(math.sqrt(intermediate_sample.shape[0])),
                    ),
                    idx,
                )

    def training_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        raise NotImplementedError(
            "Training is not supported for CounterfactualDiffusionModule"
        )

    @torch.no_grad()
    def evaluation_step(
        self, batch: BatchDict, evaluation_engine: Engine = None, **kwargs
    ) -> ModelOutput:
        input = batch[self._input_key]
        ground_truth_labels = batch[DataKeys.LABEL]
        target_query_labels = self._prepare_target_query_labels(
            input, ground_truth_labels
        )

        real_logits = self._torch_model.trainable_models.classifier(input)
        logger.info(
            f"Predicted labels: {real_logits.argmax(dim=1)}",
        )

        logger.info(
            f"Generating counterfactual samples for a batch of inputs: shape={input.shape}, min={input.min()}, max={input.max()}"
        )
        logger.info(f"Ground truth labels: {ground_truth_labels}")
        logger.info(f"Target query labels: {target_query_labels}")

        batch_done = []
        for key, gt, target in zip(
            batch["__key__"], ground_truth_labels, target_query_labels
        ):
            output_name = f"{key}_from_{gt.item()}_to_{target.item()}.png"
            possible_names = [
                Path(self._tb_logger.writer.logdir).parent / x / output_name
                for x in [
                    "class_correct_cf_correct",
                    "class_correct_cf_incorrect",
                    "class_incorrect_cf_correct",
                    "class_incorrect_cf_incorrect",
                ]
            ]
            if any([x.exists() for x in possible_names]):
                batch_done.append(key)
        if len(batch_done) > 0:
            logger.info(f"Skipping {len(batch_done)} samples that are already done")
            return {"loss": -1}

        pipeline_outputs: GuidedCounterfactualDiffusionSamplingPipelineOutput = (
            self._generate_data(
                input,
                target_query_labels,
                return_intermediate_samples=True,
                **self._prepare_model_kwargs(
                    batch,
                    stage=kwargs["stage"],
                ),
            )
        )

        # log the generated counterfactual samples
        if self._enable_tensorboard_logging:
            self._log_data_to_tensorboard(
                batch[self._input_key],
                pipeline_outputs,
                evaluation_engine.state.iteration,
                ground_truth_labels=ground_truth_labels,
                target_query_labels=target_query_labels,
            )

        logger.info(
            f"Counterfactual found for {pipeline_outputs.counterfactuals_found.sum()} samples"
        )

        return CounterfactualDiffusionModelOutput(
            sample_key=batch["__key__"],
            real_logits=real_logits,
            counterfactual_logits=pipeline_outputs.counterfactual_logits,
            real_label=ground_truth_labels,
            counterfactual_label=target_query_labels,
            real=_unnormalize(input),
            counterfactual=pipeline_outputs.counterfactuals,
        )

    def visualization_step(
        self, batch, evaluation_engine=None, training_engine=None, **kwargs
    ):
        raise NotImplementedError(
            "Visualization step is not implemented for UnconditionalDiffusionModule"
        )

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        if stage == TrainingStage.train:
            return []
        elif stage == TrainingStage.validation:
            return []
        elif stage == TrainingStage.test:
            return [self._input_key, DataKeys.LABEL]
        elif stage == TrainingStage.predict:
            return []
        else:
            raise ValueError(f"Stage {stage} not supported")
