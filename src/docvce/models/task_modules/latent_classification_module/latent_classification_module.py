# pytorch_diffusion + derived encoder decoder
# pytorch_diffusion + derived encoder decoder
import math
from functools import partial
from typing import Any, Dict, List, Optional, Union

import ignite.distributed as idist
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
from atria.models.task_modules.classification.image import ImageClassificationModule
from atria.models.task_modules.diffusion.utilities import _unnormalize
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine

logger = get_logger(__name__)


class LatentClassificationModule(ImageClassificationModule):
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
        input_key: str = DataKeys.IMAGE,
        latent_input_key: str = DataKeys.LATENT_IMAGE,
        use_precomputed_latents_if_available: bool = False,
        features_scale_factor: Optional[float] = None,
        compute_scale_factor: bool = False,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )

        # autoencoder params
        self._latent_input_key = latent_input_key
        self._compute_scale_factor = compute_scale_factor
        self._use_precomputed_latents_if_available = (
            use_precomputed_latents_if_available
        )
        self._features_scale_factor = features_scale_factor
        self._input_key = input_key

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        required_keys = (
            [self._input_key, DataKeys.LABEL]
            if not self._use_precomputed_latents_if_available
            else [self._latent_input_key, DataKeys.LABEL]
        )
        if stage == TrainingStage.train:
            return required_keys
        elif stage == TrainingStage.validation:
            return required_keys
        elif stage == TrainingStage.test:
            return required_keys
        elif stage == TrainingStage.visualization:
            return required_keys
        elif stage == TrainingStage.predict:
            return []
        elif stage == "FeatureExtractor":
            return [self._input_key]
        else:
            raise ValueError(f"Stage {stage} not supported")

    def _set_scaling_factor(self, vae, scaling_factor):
        if isinstance(vae, AutoencoderKL):
            vae.register_to_config(scaling_factor=scaling_factor)
        else:
            vae.scaling_factor = scaling_factor

    def _get_scaling_factor(self, vae):
        if isinstance(vae, AutoencoderKL):
            return vae.config.scaling_factor
        else:
            return vae.scaling_factor

    def _build_model(
        self,
    ) -> Union[torch.nn.Module, TorchModelDict]:
        model: TorchModelDict = super()._build_model()

        # make sure the underlying model got an encode and decode method
        assert hasattr(
            model.non_trainable_models, "vae"
        ), "The non_trainable_models models must contain a `vae` underlying model."
        model.non_trainable_models.vae.eval()
        model.non_trainable_models.vae.requires_grad_(False)
        if self._features_scale_factor is not None:
            self._set_scaling_factor(
                model.non_trainable_models.vae, self._features_scale_factor
            )

        return model

    def _build_model(self) -> Union[torch.nn.Module, TorchModelDict]:
        model: TorchModelDict = super()._build_model()
        assert hasattr(
            model.non_trainable_models, "vae"
        ), "The non_trainable_models models must contain a `vae` underlying model."
        model.non_trainable_models.vae.eval()
        model.non_trainable_models.vae.requires_grad_(False)
        if self._features_scale_factor is not None:
            self._set_scaling_factor(
                model.non_trainable_models.vae, self._features_scale_factor
            )
        assert hasattr(
            model.trainable_models, "classifier"
        ), "The trainable_models models must contain a `classifier` underlying model."
        return model

    @torch.no_grad()
    def compute_scale_factor(
        self, batch
    ):  # compute_scale_factor does not work correctly at the moment with resume checkpoint,
        # after computing, we need to save it in checkpoint for later runs but this does not work atm
        if (
            self._compute_scale_factor
        ):  # and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            # get data
            input = self._prepare_input(batch, scale_latents=False).detach()
            old_scaling_factor = self._get_scaling_factor(
                self._torch_model.non_trainable_models.vae
            )
            new_scaling_factor = 1.0 / input.flatten().std()
            self._set_scaling_factor(
                self._torch_model.non_trainable_models.vae, new_scaling_factor
            )
            logger.info(
                f"Using std scale factor: {new_scaling_factor} instead of checkpointed scale factor {old_scaling_factor}"
            )

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["scale_factor"] = self._get_scaling_factor(
            self._torch_model.non_trainable_models.vae
        )
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "scale_factor" in state_dict:
            self._set_scaling_factor(
                self._torch_model.non_trainable_models.vae, state_dict["scale_factor"]
            )

    def encode(self, input: torch.Tensor, scale_latents: bool = True) -> torch.Tensor:
        with torch.cuda.amp.autocast(
            enabled=False
        ):  # with fp16 forward pass on latent, we get nans
            latents = self._torch_model.non_trainable_models.vae.encode(
                input
            ).latent_dist.sample()
            if scale_latents:
                latents = latents * self._get_scaling_factor(
                    self._torch_model.non_trainable_models.vae
                )
            return latents

    @torch.no_grad()
    def _prepare_input(
        self, batch: BatchDict, scale_latents: bool = True
    ) -> torch.Tensor:
        if self._use_precomputed_latents_if_available:
            assert (
                self._latent_input_key in batch
            ), f"Key {self._latent_input_key} not found in batch. "
            latents = batch[self._latent_input_key]
            if scale_latents:
                latents = latents * self._get_scaling_factor(
                    self._torch_model.non_trainable_models.vae
                )
        else:
            input = batch[self._input_key]
            latents = self.encode(input, scale_latents=scale_latents)
        return latents

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
        kwargs = {}
        for key in self._torch_model_builder:
            if key == "classifier":
                kwargs["classifier"] = classifier_kwargs
            else:
                kwargs[key] = {}
        return kwargs

    def training_step(
        self, batch: BatchDict, training_engine: Engine, **kwargs
    ) -> ModelOutput:
        # we compute scaling factor on first input batch and first iteration
        if training_engine.state.iteration == 1 and training_engine.state.epoch == 1:
            if idist.get_rank() > 0:
                idist.barrier()
            self.compute_scale_factor(batch)

            if idist.get_rank() == 0:
                idist.barrier()
        batch[self._latent_input_key] = self._prepare_input(batch)
        return super().training_step(batch=batch, **kwargs)

    def feature_extractor_step(self, batch: BatchDict, engine, **kwargs) -> ModelOutput:
        self._use_precomputed_latents_if_available = False
        latents = self._prepare_input(batch, scale_latents=False)

        if engine.state.iteration == 1:
            logger.info("Adding feature extraction images to tensorboard")
            reconstructed = self._torch_model.non_trainable_models.vae.decode(
                latents
            ).sample
            self._tb_logger.writer.add_image(
                f"feature_extractor/{self._input_key}",
                torchvision.utils.make_grid(
                    _unnormalize(batch[self._input_key]),
                    normalize=False,
                    nrow=int(math.sqrt(batch[self._input_key].shape[0])),
                ),
                engine.state.iteration,
            )
            self._tb_logger.writer.add_image(
                f"feature_extractor/reconstructed/{self._input_key}",
                torchvision.utils.make_grid(
                    _unnormalize(reconstructed),
                    normalize=False,
                    nrow=int(math.sqrt(reconstructed.shape[0])),
                ),
                engine.state.iteration,
            )
            self._tb_logger.writer.flush()
        return {
            self._latent_input_key: latents,
        }

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        batch[self._latent_input_key] = self._prepare_input(batch)
        return super().evaluation_step(batch=batch, **kwargs)

    def _model_forward(self, batch: Dict[str, torch.Tensor | torch.Any]) -> torch.Any:
        assert (
            self._model_built
        ), "Model must be built before training. Call build_model() first"
        if isinstance(self._torch_model, dict):
            raise NotImplementedError(
                "Model forward must be implemented in the task module when multiple models are used"
            )
        return self._torch_model.trainable_models.classifier(
            batch[self._latent_input_key]
        )
