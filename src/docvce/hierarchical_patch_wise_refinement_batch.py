import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from PIL import Image
from ray import logger
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from docvce.models.dit_model import DitModel

logging.basicConfig(level=logging.INFO)


class CounterfactualDataset(Dataset):
    def __init__(self, cf_image_paths: List[Path], real_samples_dir: Path):
        self.cf_image_paths = cf_image_paths
        self.real_samples_dir = real_samples_dir

    def __len__(self):
        return len(self.cf_image_paths)

    def __getitem__(self, idx):
        cf_image_path = self.cf_image_paths[idx]
        real_image_path = self.real_samples_dir / "class_correct" / cf_image_path.name
        if not real_image_path.exists():
            real_image_path = (
                self.real_samples_dir / "class_incorrect" / cf_image_path.name
            )

        cf_image = load_image(cf_image_path)
        real_image = load_image(real_image_path)
        return cf_image, real_image, cf_image_path.name


def get_prediction(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    return model(image.to(0)).argmax(-1)


def get_prediction_probs(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(model(image.to(0)), dim=-1)


def replace_patch(
    image: torch.Tensor, real_image: torch.Tensor, x: int, y: int, patch_size: int
) -> torch.Tensor:
    modified_image = image.clone()
    modified_image[:, y : y + patch_size, x : x + patch_size] = real_image[
        :, y : y + patch_size, x : x + patch_size
    ]
    return modified_image


def get_current_replacement_image_and_prediction(
    model: torch.nn.Module,
    current_images: torch.Tensor,
    real_images: torch.Tensor,
    curr_patches_batch: List[Tuple[int, int, int]],
    target_prediction_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Identify indices that need patch replacement
    patch_exists = torch.tensor([p is not None for p in curr_patches_batch])
    current_modified_images = current_images.clone()

    # If no patches to modify, return early
    if not patch_exists.any():
        return current_modified_images, target_prediction_probs.clone()

    # Extract valid patches and their corresponding indices
    valid_indices = torch.nonzero(patch_exists, as_tuple=True)[0]
    valid_patches = [curr_patches_batch[i] for i in valid_indices]

    # Apply patch replacement in a vectorized fashion
    for idx, (h, w, size) in zip(valid_indices, valid_patches):
        current_modified_images[idx, :, h : h + size, w : w + size] = real_images[
            idx, :, h : h + size, w : w + size
        ]

    # Get predictions for the modified images (only those with changes)
    modified_images = current_modified_images[valid_indices]

    processed_batch_probs = get_prediction_probs(model, modified_images)
    new_prediction_probs = target_prediction_probs.clone()
    new_prediction_probs[valid_indices] = processed_batch_probs
    return current_modified_images, new_prediction_probs


def hierarchical_patch_wise_refinement(
    cf_images: torch.Tensor,
    real_images: torch.Tensor,
    model: torch.nn.Module,
    cf_targets,
    initial_grid_size: int = 2,
    min_patch_size: int = 16,
    confidence_diff_threshold: float = 0.1,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    batch_size, _, h, w = cf_images.shape
    current_images = cf_images.clone()
    patch_size = min(h, w) // initial_grid_size
    patches_per_image = [
        [
            (x, y, patch_size)
            for y in range(0, h, patch_size)
            for x in range(0, w, patch_size)
        ]
        for _ in range(batch_size)
    ]
    target_prediction_probs = get_prediction_probs(model, current_images)
    target_prediction = target_prediction_probs.argmax(-1)
    if not all(target_prediction.cpu() == torch.tensor(cf_targets)):
        logger.warning(
            f"Target prediction {target_prediction} != cf_targets {cf_targets}"
        )

    pbar = tqdm.tqdm()
    while any([len(x) > 0 for x in patches_per_image]):
        assert (
            len(patches_per_image) == batch_size
        ), "Patches per image should be equal to batch size"
        curr_patches_batch = [
            x.pop(0) if len(x) > 0 else None for x in patches_per_image
        ]
        current_modified_images, new_prediction_probs = (
            get_current_replacement_image_and_prediction(
                model=model,
                current_images=current_images,
                real_images=real_images,
                curr_patches_batch=curr_patches_batch,
                target_prediction_probs=target_prediction_probs,
            )
        )
        for idx in range(batch_size):
            if (new_prediction_probs[idx].argmax(-1) == target_prediction[idx]) and (
                torch.abs(
                    target_prediction_probs[idx][target_prediction[idx]]
                    - new_prediction_probs[idx][target_prediction[idx]]
                )
                < confidence_diff_threshold
            ):
                current_images[idx] = current_modified_images[idx]
            else:
                x, y, patch_size = curr_patches_batch[idx]
                half_size = patch_size // 2
                if half_size >= min_patch_size:
                    patches_per_image[idx].extend(
                        [
                            (x, y, half_size),
                            (x + half_size, y, half_size),
                            (x, y + half_size, half_size),
                            (x + half_size, y + half_size, half_size),
                        ]
                    )
        pbar.update(1)
    pbar.close()
    final_prediction_probs = get_prediction_probs(model, current_images)
    print(
        "Final prediction probs: ",
        final_prediction_probs[torch.arange(batch_size), target_prediction],
    )
    final_prediction = final_prediction_probs.argmax(-1)
    if not all(  # this is not true in case all search is exhausted and all patches in last iteration are replaced
        final_prediction == (target_prediction)
    ):
        logger.warning(
            f"Start prediction {target_prediction} != Final prediction {final_prediction}"
        )
    diff_images = torch.abs(current_images - real_images)
    diff_images = diff_images / diff_images.max()
    return current_images, diff_images


def setup_model(args: argparse.Namespace) -> None:
    from timm import create_model

    checkpoint = args.model_checkpoint
    checkpoint = torch.load(checkpoint)
    if args.model_type == "convnext_base":
        checkpoint = {
            k.replace("model.", ""): v
            for k, v in checkpoint["task_module"]["model"].items()
        }
        model = create_model(
            "convnext_base", pretrained=False, num_classes=args.num_classes
        )
    elif args.model_type == "resnet50":
        checkpoint = {
            k.replace("model.", ""): v
            for k, v in checkpoint["task_module"]["model"].items()
        }
        model = create_model("resnet50", pretrained=False, num_classes=args.num_classes)
    elif args.model_type == "dit_b":
        checkpoint = {k: v for k, v in checkpoint["task_module"]["model"].items()}
        model = DitModel(num_labels=args.num_classes)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(0)
    return model


def setup_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    experiment_dir = Path(args.experiment_dir)
    real_samples_dir = experiment_dir / "real_samples"

    run_dirs = []
    for dir in os.listdir(experiment_dir):
        if "refined" in dir or "refined3" in dir:
            continue
        if os.path.isdir(experiment_dir / dir) and not dir in [
            "real_samples",
            "real_samples_split_2",
        ]:
            run_dirs.append(experiment_dir / dir)
    logger.info(f"Found total {len(run_dirs)} runs: {run_dirs}")
    return run_dirs, real_samples_dir


def load_image(image_path: str) -> torch.Tensor:
    return (to_tensor(Image.open(image_path).convert("RGB")) - 0.5) / 0.5


@torch.no_grad()
def main(args) -> None:
    model = setup_model(args)
    run_dirs, real_samples_dir = setup_paths(args)
    target_run_index = list(
        range(
            int(args.target_run_index.split("-")[0]),
            int(args.target_run_index.split("-")[1]),
        )
    )
    logger.info(f"Processing runs: {target_run_index}")
    for run_idx, run_dir in enumerate(run_dirs):
        if run_idx not in target_run_index:
            continue
        cf_image_paths = list(
            (run_dir / "class_correct_cf_correct").glob("*.png")
        ) + list((run_dir / "class_incorrect_cf_correct").glob("*.png"))

        dataset = CounterfactualDataset(cf_image_paths, real_samples_dir)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
        )

        output_base_path = (
            run_dir.parent.parent / (run_dir.parent.name + "-refined3") / run_dir.name
        )
        output_base_path_diff = (
            run_dir.parent.parent / (run_dir.parent.name + "-diff3") / run_dir.name
        )
        if output_base_path.exists() and len(
            list(output_base_path.glob("*.png"))
        ) == len(dataset):
            logger.info(
                f"Already processed images: {output_base_path}, Dataset: {len(dataset)}, Images: {len(list(output_base_path.glob('*.png')))}"
            )
            continue

        output_base_path.mkdir(parents=True, exist_ok=True)
        output_base_path_diff.mkdir(parents=True, exist_ok=True)

        for idx, (cf_images, real_images, image_names) in tqdm.tqdm(
            enumerate(dataloader), desc=f"Processing Images for {run_dir.name}"
        ):
            output_image_paths = [output_base_path / name for name in image_names]
            cf_targets = [
                int(name.replace(".png", "").split("to_")[1]) for name in image_names
            ]
            if all([x.exists() for x in output_image_paths]):
                continue
            logging.info(f"Cf targets: {cf_targets}")
            current_images, diff_images = hierarchical_patch_wise_refinement(
                cf_images, real_images, model, cf_targets
            )

            if args.save_diffs or idx == 0:
                for number_diffs, (img, output_image_path) in enumerate(
                    zip(diff_images, output_image_paths)
                ):
                    output_image_path_diff = (
                        output_base_path_diff / output_image_path.name
                    )
                    save_image(img, output_image_path_diff)
                    if number_diffs > 32:
                        break

            print("Output dir: ", output_image_paths[0])
            for img, output_image_path, cf_target in zip(
                current_images, output_image_paths, cf_targets
            ):
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                save_image((img.cpu() * 0.5 + 0.5).clamp(0, 1), output_image_path)
                # image = Image.open(output_image_path).convert("RGB")
                # image = (to_tensor(image) - 0.5) / 0.5
                # image = image.unsqueeze(0)
                # prediction = get_prediction(model, image)
                # print(f"Prediction: {prediction}, Target: {cf_target}")


def arguments():
    parser = argparse.ArgumentParser(description="Hierarchical Patchwise Refinement")
    parser.add_argument(
        "--experiment_dir", required=True, type=str, help="Path to real images"
    )
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        help="Model type to use for processing",
    )
    parser.add_argument(
        "--num_classes", required=True, type=int, help="Number of classes for the model"
    )
    parser.add_argument(
        "--model_checkpoint",
        required=True,
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for processing"
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Debug mode for testing"
    )
    parser.add_argument(
        "--save_diffs",
        type=bool,
        default=False,
        help="Save difference images between real and counterfactual",
    )
    parser.add_argument(
        "--target_run_index",
        type=str,
        default="0-10",
        help="Run index to process (e.g., '0-10' for range or '0,2,5' for specific indices)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arguments())
