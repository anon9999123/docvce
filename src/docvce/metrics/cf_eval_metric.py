from dataclasses import dataclass
from pathlib import Path

import torch
from ignite.engine import Engine
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.utils import apply_to_tensor
from torchvision.transforms.functional import to_pil_image

from docvce.models.task_modules.diffusion.counterfactual_diffusion import (
    CounterfactualDiffusionModelOutput,
)


def identity(x):
    return x


@dataclass
class ScoresDict:
    cf_inv_conf_score: float = 0.0
    l1_norm: float = 0.0
    l2_norm: float = 0.0
    num_samples: int = 0


class CounterfactualEvaluationMetric(Metric):
    def __init__(
        self,
        real_output_path: str,
        cf_output_path: str,
        output_transform=identity,
        device="cpu",
    ):
        self._summary = {
            "class_correct_cf_correct": ScoresDict(),
            "class_correct_cf_incorrect": ScoresDict(),
            "class_correct": ScoresDict(),
            "class_incorrect_cf_correct": ScoresDict(),
            "class_incorrect_cf_incorrect": ScoresDict(),
            "class_incorrect": ScoresDict(),
            "clean_accuracy": 0,
            "counterfactual_accuracy": 0,
            "total": ScoresDict(),
        }
        self._real_output_path = Path(real_output_path)
        self._cf_output_path = Path(cf_output_path)
        super(CounterfactualEvaluationMetric, self).__init__(
            output_transform=output_transform, device=device
        )

    def save_real_images(
        self, image_name: str, real: torch.Tensor, class_correct: bool
    ) -> None:
        output_path = self._real_output_path
        output_path = (
            self._real_output_path / "class_correct"
            if class_correct
            else self._real_output_path / "class_incorrect"
        )
        output_path.mkdir(parents=True, exist_ok=True)
        real_pil = to_pil_image(real)
        if not (output_path / image_name).exists():
            real_pil.save(output_path / image_name)

    def save_cf_images(
        self,
        image_name: str,
        counterfactual: torch.Tensor,
        class_correct: bool,
        cf_correct: bool,
    ) -> None:
        output_path = self._cf_output_path
        dir_name = "class_correct" if class_correct else "class_incorrect"
        dir_name += "_cf_correct" if cf_correct else "_cf_incorrect"
        output_path = output_path / dir_name
        output_path.mkdir(parents=True, exist_ok=True)
        cf_pil = to_pil_image(counterfactual)
        if not (output_path / image_name).exists():
            cf_pil.save(output_path / image_name)

    def save_info(
        self,
        image_name: str,
        class_correct: bool,
        cf_correct: bool,
        ground_truth_label: int,
        real_prediction: int,
        counterfactual_label: int,
        counterfactual_prediction: int,
        cf_inv_conf_score: float,
        l_1: float,
        l_2: float,
    ) -> None:
        output_path = self._cf_output_path
        dir_name = "class_correct" if class_correct else "class_incorrect"
        dir_name += "_cf_correct" if cf_correct else "_cf_incorrect"
        output_path = output_path / dir_name
        output_path.mkdir(parents=True, exist_ok=True)
        info_output_path = (output_path / image_name).with_suffix(".txt")
        to_write = (
            f"label: {ground_truth_label}"
            + f"\npred: {real_prediction}"
            + f"\ntarget: {counterfactual_label}"
            + f"\ncf pred: {counterfactual_prediction}"
            + f"\ncf_inv_conf_score: {cf_inv_conf_score}"
            + f"\nl_1: {l_1}"
            + f"\nl_2: {l_2}"
        )
        with open(info_output_path, "w") as f:
            f.write(to_write)

    def save_sample_image(
        self,
        real,
        counterfactual,
        class_correct,
        cf_correct,
        sample_key,
        ground_truth_label: int,
        counterfactual_label: int,
        real_prediction: int,
        counterfactual_prediction: int,
        cf_inv_conf_score: float,
        l_1: float,
        l_2: float,
    ):
        # save real images
        image_name = f"{sample_key}_from_{ground_truth_label}_to_{str(counterfactual_label.item())}.png"
        self.save_real_images(image_name, real, class_correct)

        # save cf images
        self.save_cf_images(image_name, counterfactual, class_correct, cf_correct)

        # save info
        self.save_info(
            image_name,
            class_correct,
            cf_correct,
            ground_truth_label,
            real_prediction,
            counterfactual_label,
            counterfactual_prediction,
            cf_inv_conf_score,
            l_1,
            l_2,
        )

    @reinit__is_reduced
    def reset(self):
        self._summary = {
            "class_correct_cf_correct": ScoresDict(),
            "class_correct_cf_incorrect": ScoresDict(),
            "class_correct": ScoresDict(),
            "class_incorrect_cf_correct": ScoresDict(),
            "class_incorrect_cf_incorrect": ScoresDict(),
            "class_incorrect": ScoresDict(),
            "clean_accuracy": 0,
            "counterfactual_accuracy": 0,
            "total": ScoresDict(),
        }
        super(CounterfactualEvaluationMetric, self).reset()

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)

    @reinit__is_reduced
    def update(self, output: CounterfactualDiffusionModelOutput):
        if not isinstance(output, CounterfactualDiffusionModelOutput):
            return
        class_correct = output.real_label == output.real_logits.argmax(-1)
        cf_correct = output.counterfactual_label == output.counterfactual_logits.argmax(
            -1
        )

        # find conf scores
        counterfactual_probs = torch.softmax(output.counterfactual_logits, dim=1)
        conf_scores = counterfactual_probs[
            range(counterfactual_probs.size(0)), output.counterfactual_label.view(-1)
        ]
        cf_inv_conf_score = 1 - conf_scores  # this is the same as BKL in dime paper

        # find the l1 distance between the input and counterfactual samples
        # _plot_tensors(output.real)
        # _plot_tensors(output.counterfactual)
        bsz = output.real.shape[0]
        diff = output.real.view(bsz, -1) - output.counterfactual.view(bsz, -1)
        l1_norm = torch.norm(diff, p=1, dim=-1)
        l2_norm = torch.norm(diff, p=2, dim=-1)

        # detach all tensors
        for tensor in [
            counterfactual_probs,
            cf_inv_conf_score,
            l1_norm,
        ]:
            apply_to_tensor(tensor, lambda tensor: tensor.detach().cpu())

        # update summary
        for k, cond in zip(
            [
                "class_correct_cf_correct",
                "class_correct_cf_incorrect",
                "class_correct",
                "class_incorrect_cf_correct",
                "class_incorrect_cf_incorrect",
                "class_incorrect",
                "total",
            ],
            [
                class_correct & cf_correct,
                class_correct & ~cf_correct,
                class_correct,
                ~class_correct & cf_correct,
                ~class_correct & ~cf_correct,
                ~class_correct,
            ],
        ):
            if all(cond == False):
                continue
            num_samples = torch.count_nonzero(cond).item()
            cond = cond.detach().cpu()
            self._summary[k].l1_norm += l1_norm[cond].sum()
            self._summary[k].l2_norm += l2_norm[cond].sum()
            self._summary[k].num_samples += num_samples

        for sample_idx in range(output.real.shape[0]):
            self.save_sample_image(
                output.real[sample_idx],
                output.counterfactual[sample_idx],
                class_correct[sample_idx],
                cf_correct[sample_idx],
                sample_key=output.sample_key[sample_idx],
                ground_truth_label=output.real_label[sample_idx],
                counterfactual_label=output.counterfactual_label[sample_idx],
                real_prediction=output.real_logits.argmax(-1)[sample_idx],
                counterfactual_prediction=output.counterfactual_logits.argmax(-1)[
                    sample_idx
                ],
                cf_inv_conf_score=cf_inv_conf_score[sample_idx],
                l_1=l1_norm[sample_idx],
                l_2=l2_norm[sample_idx],
            )

    @sync_all_reduce("_summary")
    def compute(self):
        for k in [
            "class_correct_cf_correct",
            "class_correct_cf_incorrect",
            "class_correct",
            "class_incorrect_cf_correct",
            "class_incorrect_cf_incorrect",
            "class_incorrect",
            "total",
        ]:
            if self._summary[k].num_samples > 0:
                self._summary[k].l2_norm /= self._summary[k].num_samples
                self._summary[k].l1_norm /= self._summary[k].num_samples
        return self._summary
