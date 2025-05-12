import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from ray import logger
from torchvision.transforms.functional import gaussian_blur, to_tensor

logging.basicConfig(level=logging.INFO)

colors = []
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
transparent_red = LinearSegmentedColormap.from_list("transparent_red", colors)

colors = []
for j in np.linspace(0, 1, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
transparent_blue = LinearSegmentedColormap.from_list("transparent_blue", colors)


def get_prediction(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    image = image.to(0)
    return model(image.unsqueeze(0)).argmax(-1)


def replace_patch(
    image: torch.Tensor, real_image: torch.Tensor, x: int, y: int, patch_size: int
) -> torch.Tensor:
    modified_image = image.clone()
    modified_image[:, y : y + patch_size, x : x + patch_size] = real_image[
        :, y : y + patch_size, x : x + patch_size
    ]
    return modified_image


def replace_patch_with_smooth_boundaries(
    image: torch.Tensor,
    real_image: torch.Tensor,
    x: int,
    y: int,
    patch_size: int,
    blend_radius: int = 8,
) -> torch.Tensor:
    modified_image = image.copy()
    blended_patch_size = patch_size + blend_radius * 2
    mask = torch.zeros((blended_patch_size, blended_patch_size), dtype=torch.float32)
    mask[
        int(blend_radius) : -int(blend_radius), int(blend_radius) : -int(blend_radius)
    ] = 1.0
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = gaussian_blur(
        mask, (blended_patch_size // 2 + 1, blended_patch_size // 2 + 1)
    )
    mask = mask.squeeze(0).squeeze(0)
    mask = mask / mask.max()
    mask_expanded = mask.unsqueeze(0)
    start_y = y - blend_radius
    start_x = x - blend_radius
    end_y = y + patch_size + blend_radius
    end_x = x + patch_size + blend_radius

    if start_y < 0:
        start_y = 0
        mask_expanded = mask_expanded[:, blend_radius:, :]
    if start_x < 0:
        start_x = 0
        mask_expanded = mask_expanded[:, :, blend_radius:]
    if end_y > image.shape[0]:
        end_y = image.shape[0]
        mask_expanded = mask_expanded[:, :-blend_radius, :]
    if end_x > image.shape[1]:
        end_x = image.shape[1]
        mask_expanded = mask_expanded[:, :, :-blend_radius]

    modified_image[:, start_y:end_y, start_x:end_x] = (
        modified_image[:, start_y:end_y, start_x:end_x] * (1 - mask_expanded)
        + real_image[:, start_y:end_y, start_x:end_x] * mask_expanded
    )

    return modified_image


@torch.no_grad
def binary_patchwise_replacement(
    cf_image: torch.Tensor,
    real_image: torch.Tensor,
    model: torch.nn.Module,
    initial_grid_size: int = 2,
    min_patch_size: int = 16,
    debug: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    c, h, w = cf_image.shape
    current_image = cf_image.clone()
    current_prediction = get_prediction(model, current_image)
    patch_size = min(h, w) // initial_grid_size
    patches_to_refine = [
        (x, y, patch_size)
        for y in range(0, h, patch_size)
        for x in range(0, w, patch_size)
    ]

    draw_image = None
    if debug:
        draw_image = current_image.clone().permute(1, 2, 0).cpu().numpy().copy()

    while patch_size >= min_patch_size:
        next_patches = []

        for x, y, _ in patches_to_refine:
            modified_image = replace_patch(current_image, real_image, x, y, patch_size)
            new_prediction = get_prediction(model, modified_image)

            if new_prediction == current_prediction:
                current_image = modified_image
            else:
                if debug:
                    cv2.rectangle(
                        draw_image,
                        (x, y),
                        (x + patch_size, y + patch_size),
                        (255, 0, 0),
                        2,
                    )
                if patch_size // 2 >= min_patch_size:
                    half_size = patch_size // 2
                    next_patches.extend(
                        [
                            (x, y, half_size),
                            (x + half_size, y, half_size),
                            (x, y + half_size, half_size),
                            (x + half_size, y + half_size, half_size),
                        ]
                    )

        if not next_patches:
            break

        patches_to_refine = next_patches
        patch_size //= 2

    return current_image, draw_image


def plot_image(image: torch.Tensor, title: str = "", ax=None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax.set_title(title)
    ax.axis("off")
    return ax


def generate_difference_map(
    real_image: torch.Tensor, current_image: torch.Tensor
) -> torch.Tensor:
    diff_image = torch.abs(current_image - real_image).mean(dim=0).cpu().numpy()
    diff_image = diff_image / (diff_image.max() + 1e-8)  # Normalize
    colormap = transparent_red(diff_image)
    colormap = (colormap * 255).astype(np.uint8)
    difference_map = Image.fromarray(colormap, mode="RGBA")
    background = Image.new(
        "RGB", difference_map.size, (255, 255, 255)
    )  # White background
    difference_map = Image.alpha_composite(
        background.convert("RGBA"), difference_map
    ).convert("RGB")
    return to_tensor(difference_map)


def setup_model(args: argparse.Namespace) -> None:
    from timm import create_model

    checkpoint = (
        args.model_checkpoint
    )  # "./pretrained_models/convnext_base_rvlcdip_basic.pt"
    checkpoint = torch.load(checkpoint)
    checkpoint = {
        k.replace("model.", ""): v
        for k, v in checkpoint["task_module"]["model"].items()
    }
    model = create_model("convnext_base", pretrained=False, num_classes=16)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(0)
    return model


def setup_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    experiment_dir = Path(args.experiment_dir)
    real_samples_dir = experiment_dir / "real_samples"

    run_dirs = []
    for dir in os.listdir(experiment_dir):
        if "refined" in dir:
            continue
        if os.path.isdir(experiment_dir / dir) and not dir in [
            "real_samples",
        ]:
            run_dirs.append(experiment_dir / dir)
    logger.info(f"Found total {len(run_dirs)} runs: {run_dirs}")
    return run_dirs, real_samples_dir


def load_image(image_path: str) -> torch.Tensor:
    return (to_tensor(Image.open(image_path).convert("RGB")) - 0.5) / 0.5


def refine_counterfactual(
    model: torch.nn.Module,
    cf_image: torch.Tensor,
    real_image: torch.Tensor,
    cf_target: int,
    debug: bool = False,
) -> None:
    start_prediction = get_prediction(model, cf_image)
    assert start_prediction == int(
        cf_target
    ), f"Start prediction {start_prediction} != Target {cf_target}"
    current_image, draw_image = binary_patchwise_replacement(
        cf_image, real_image, model, debug=debug
    )
    final_prediction = get_prediction(model, current_image)
    assert final_prediction == int(
        start_prediction
    ), f"Start prediction {start_prediction} != Final prediction {final_prediction}"
    difference_map = generate_difference_map(real_image, current_image)
    return current_image, draw_image, difference_map


def save_images(
    output_image_path: str,
    cf_image: str,
    current_image: str,
    real_image: str,
    difference_map: str,
    draw_image: str,
    debug: bool,
) -> None:
    if debug:
        draw_image = torch.from_numpy(draw_image).permute(2, 0, 1)
        grid = torch.stack(
            [
                cf_image,
                current_image,
                real_image,
                difference_map,
                draw_image,
            ]
        )
    else:
        grid = torch.stack([cf_image, current_image, real_image, difference_map])
    grid = (grid * 0.5) + 0.5  # Denormalize the images
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    titles = [
        "Counterfactual Image",
        "Refined Image",
        "Real Image",
        "Difference Map",
    ]
    if debug:
        titles.append("Debug Image")
    for ax, image, title in zip(axes.flatten(), grid, titles):
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(title)
        ax.axis("off")
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, bbox_inches="tight")
    plt.close(fig)

    Image.fromarray(
        (cf_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ).save(output_image_path.with_name(output_image_path.stem + "_cf.png"))
    Image.fromarray(
        (current_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ).save(output_image_path.with_name(output_image_path.stem + "_current.png"))
    Image.fromarray(
        (real_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ).save(output_image_path.with_name(output_image_path.stem + "_real.png"))
    Image.fromarray(
        (difference_map.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ).save(output_image_path.with_name(output_image_path.stem + "_difference.png"))


@torch.no_grad()
def main() -> None:
    args = arguments()
    model = setup_model(args)
    run_dirs, real_samples_dir = setup_paths(args)
    for run_dir in run_dirs:
        cf_image_paths = list(
            (run_dir / "class_correct_cf_correct").glob("*.png")
        ) + list((run_dir / "class_incorrect_cf_correct").glob("*.png"))

        pbar = tqdm.tqdm(cf_image_paths, desc=f"Processing Images for {run_dir.name}")
        for cf_image_path in tqdm.tqdm(cf_image_paths, desc="Processing Images"):
            output_image_path = (
                run_dir.parent / (run_dir.name + "-refined") / cf_image_path.name
            )
            # if output_image_path.exists():
            #     continue
            pbar.set_postfix({"image": cf_image_path.name})
            cf_target = cf_image_path.name.split("to_")[1].split(".")[0]
            real_image_path = real_samples_dir / "class_correct" / cf_image_path.name
            if not real_image_path.exists():
                real_image_path = (
                    real_samples_dir / "class_incorrect" / cf_image_path.name
                )

            cf_image = load_image(cf_image_path)
            real_image = load_image(real_image_path)
            current_image, draw_image, difference_map = refine_counterfactual(
                model, cf_image, real_image, cf_target, debug=args.debug
            )
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            # save_image(current_image, output_image_path)
            save_images(
                run_dir.parent / (run_dir.name + "-refined") / cf_image_path.name,
                cf_image,
                current_image,
                real_image,
                difference_map,
                draw_image,
                args.debug,
            )


def arguments():
    parser = argparse.ArgumentParser(description="Hierarchical Patchwise Refinement")
    parser.add_argument(
        "--experiment_dir", required=True, type=str, help="Path to real images"
    )
    parser.add_argument(
        "--model_checkpoint",
        required=True,
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="Debug mode for testing"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
