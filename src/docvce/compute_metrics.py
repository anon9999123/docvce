import argparse
import logging
import os
import warnings
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def _plot_tensors(x: torch.Tensor, name: str = ""):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    x = make_grid(x, nrow=int(x.shape[0] ** 0.5), normalize=False)
    fig, ax = plt.figure(), plt.gca()
    ax.imshow(x.detach().cpu().permute(1, 2, 0).numpy())
    ax.set_title(name)
    plt.show()


class CounterfactualDataset(data.Dataset):
    def __init__(self, real_images_dir: str, counterfactual_images_dir: str):
        # set up the paths
        self.real_images_dir = Path(real_images_dir)
        self.counterfactual_images_dir = Path(counterfactual_images_dir)
        self._real_image_paths = []
        self._counterfactual_image_paths = []
        for cf_path in ["class_correct_cf_correct", "class_incorrect_cf_correct"]:
            cf_path = counterfactual_images_dir / cf_path
            for img_file in cf_path.glob("*.png"):
                self._counterfactual_image_paths.append(img_file)
                self._real_image_paths.append(
                    (real_images_dir / "class_correct" / img_file.name)
                    if cf_path.name == "class_correct_cf_correct"
                    else (real_images_dir / "class_incorrect" / img_file.name)
                )
                assert self._real_image_paths[
                    -1
                ].exists(), f"{self._real_image_paths[-1]} does not exist"
                assert self._counterfactual_image_paths[
                    -1
                ].exists(), f"{self._counterfactual_image_paths[-1]} does not exist"

        # assert lengths are same
        assert len(self._real_image_paths) == len(
            self._counterfactual_image_paths
        ), "Lengths of real and counterfactual images are not same"

        # set up the transformation
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self._real_image_paths)

    def __getitem__(self, idx):
        real_img = self.load_img(self._real_image_paths[idx])
        counterfactual_img = self.load_img(self._counterfactual_image_paths[idx])
        return real_img, counterfactual_img

    def load_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        return self._transform(img)


class CounterfactualInfoDataset(data.Dataset):
    def __init__(self, counterfactual_infos_dir: str):
        self._counterfactual_info_paths = []
        for cf_path in [
            "class_correct_cf_correct",
            "class_incorrect_cf_correct",
            "class_correct_cf_incorrect",
            "class_incorrect_cf_incorrect",
        ]:
            cf_path = counterfactual_infos_dir / cf_path
            for info_file in cf_path.glob("*.txt"):
                self._counterfactual_info_paths.append(info_file)

    def __len__(self):
        return len(self._counterfactual_info_paths)

    def __getitem__(self, idx):
        with open(self._counterfactual_info_paths[idx], "r") as f:
            info = yaml.safe_load(f)
        return info


class UnpairedCounterfactualDataset(CounterfactualDataset):
    def __init__(self, real_images_dir: str, counterfactual_images_dir: str):
        # set up the paths
        self.real_images_dir = Path(real_images_dir)
        self.counterfactual_images_dir = Path(counterfactual_images_dir)
        self._real_image_paths = []
        self._counterfactual_image_paths = []
        for cf_path in ["class_correct_cf_correct", "class_incorrect_cf_correct"]:
            cf_path = counterfactual_images_dir / cf_path
            for img_file in cf_path.glob("*.png"):
                self._counterfactual_image_paths.append(img_file)

        for img_file in real_images_dir.glob("*.png"):
            self._real_image_paths.append(img_file)

        # get real images equal to counterfactual images
        self._real_image_paths = self._real_image_paths[
            : len(self._counterfactual_image_paths)
        ]

        # set up the transformation
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self._real_image_paths)

    def __getitem__(self, idx):
        real_img = self.load_img(self._real_image_paths[idx])
        counterfactual_img = self.load_img(self._counterfactual_image_paths[idx])
        return real_img, counterfactual_img

    def load_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        return self._transform(img)


def get_activations(dataloader, model, batch_elem_idx, dims=2048, device="cuda:0"):
    model.eval()
    pred_arr = np.empty((len(dataloader.dataset), dims))
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch[batch_elem_idx]
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        # _plot_tensors(batch, "batch")
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    return pred_arr


def compute_statistics_of_path(dataloader, model, batch_elem_idx, dims, device):
    act = get_activations(
        dataloader=dataloader,
        model=model,
        batch_elem_idx=batch_elem_idx,
        dims=dims,
        device=device,
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_fid(dataloader, dims=2048, device: str = "cuda:0"):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = compute_statistics_of_path(
        dataloader=dataloader, model=model, batch_elem_idx=0, dims=dims, device=device
    )
    m2, s2 = compute_statistics_of_path(
        dataloader=dataloader, model=model, batch_elem_idx=1, dims=dims, device=device
    )
    return calculate_frechet_distance(m1, s1, m2, s2)


def arguments():
    parser = argparse.ArgumentParser(description="FVA arguments.")
    parser.add_argument(
        "--experiment_dir", required=True, type=str, help="Path to real images"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()

    experiment_dir = Path(args.experiment_dir)
    if "-refined" in experiment_dir.name:
        real_samples_split_1_dir_path = (
            experiment_dir.parent
            / experiment_dir.name.replace("-refined", "")
            / "real_samples"
        )
        real_samples_split_2_dir_path = (
            experiment_dir.parent
            / experiment_dir.name.replace("-refined", "")
            / "real_samples_split_2"
        )
    else:
        real_samples_split_1_dir_path = experiment_dir / "real_samples"
        real_samples_split_2_dir_path = experiment_dir / "real_samples_split_2"

    run_dirs = []
    for dir in os.listdir(experiment_dir):
        if os.path.isdir(experiment_dir / dir) and not dir in [
            "real_samples",
            "real_samples_split_2",
        ]:
            run_dirs.append(experiment_dir / dir)

    all_metrics = []
    with torch.no_grad():
        lpips_model = lpips.LPIPS(net="vgg").to(0)
        for run_dir in run_dirs:
            if "refined" in run_dir.name:
                continue
            if (run_dir / "metrics.csv").exists():
                prev_run = pd.read_csv(run_dir / "metrics.csv")
                if "mean_conf_score" in prev_run.columns:
                    logging.info(f"Skipping {run_dir} as metrics already computed")
                    all_metrics.append(pd.read_csv(run_dir / "metrics.csv"))
                    continue
                # continue

            results = []
            logging.info(f"Run dir: {run_dir}")
            info_dataset = CounterfactualInfoDataset(run_dir)
            dataloader_info = data.DataLoader(
                info_dataset,
                batch_size=256,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True,
                collate_fn=lambda x: x,
            )

            logging.info("Computing counterfactual info")
            n_counterfactuals_found = 0
            mean_conf_score = 0
            total_samples = 0
            for info_batch in tqdm(dataloader_info):
                for info in info_batch:
                    target = info["target"]
                    cf_pred = info["cf pred"]
                    cf_inv_conf_score = info["cf_inv_conf_score"]
                    if target == cf_pred:
                        n_counterfactuals_found += 1
                        mean_conf_score += (
                            1 - cf_inv_conf_score
                        )  # take mean conf score only for flipped samples
                    total_samples += 1
            mean_conf_score /= n_counterfactuals_found
            flip_ratio = n_counterfactuals_found / total_samples
            logging.info(f"Flip ratio: {flip_ratio}")
            logging.info(f"Mean CF inv conf score: {mean_conf_score}")
            results.append(
                {
                    "run_dir": run_dir.name,
                    "flip_ratio": flip_ratio,
                    "mean_conf_score": mean_conf_score,
                }
            )

            dataset_real_samples_split_1 = CounterfactualDataset(
                real_samples_split_1_dir_path, run_dir
            )
            dataset_real_samples_split_2 = UnpairedCounterfactualDataset(
                real_samples_split_2_dir_path, run_dir
            )
            results.append(
                {
                    "run_dir": run_dir.name,
                    "size_info_dataset": len(info_dataset),
                    "size_dataset_real_samples_split_2": len(
                        dataset_real_samples_split_2
                    ),
                    "size_dataset_real_samples_split_1": len(
                        dataset_real_samples_split_1
                    ),
                }
            )
            dataloader_split_1 = data.DataLoader(
                dataset_real_samples_split_1,
                batch_size=256,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True,
            )

            logging.info("Computing closeness metrics")
            for real, cf in tqdm(dataloader_split_1):
                real = real.to(0, dtype=torch.float)
                cf = cf.to(0, dtype=torch.float)
                bsz = real.shape[0]
                diff = real.view(bsz, -1) - cf.view(bsz, -1)
                l1_norm_sum = torch.norm(diff, p=1, dim=-1)
                l2_norm_sum = torch.norm(diff, p=2, dim=-1)
                l1_norm_mean = l1_norm_sum / diff.shape[-1]
                l2_norm_mean = l2_norm_sum / diff.shape[-1]
                lpips_loss = lpips_model(real, cf, normalize=True)

                for i in range(bsz):
                    results.append(
                        {
                            "run_dir": run_dir.name,
                            "l1_norm_sum": l1_norm_sum[i].item(),
                            "l2_norm_sum": l2_norm_sum[i].item(),
                            "l1_norm_mean": l1_norm_mean[i].item(),
                            "l2_norm_mean": l2_norm_mean[i].item(),
                            "lpips_loss": lpips_loss[i].item(),
                        }
                    )

            logging.info("Computing FID")
            fid = compute_fid(dataloader_split_1)
            dataloader_split_2 = data.DataLoader(
                dataset_real_samples_split_2,
                batch_size=256,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True,
            )

            logging.info("Computing sFID")
            sfid = compute_fid(dataloader_split_2)
            results.append({"run_dir": run_dir.name, "fid": fid, "sfid": sfid})
            df = pd.DataFrame(results)
            df = df.groupby("run_dir").mean().reset_index()
            df.to_csv(run_dir / "metrics.csv", index=False)
            logging.info(df)
            all_metrics.append(pd.read_csv(run_dir / "metrics.csv"))

    # Save results to a CSV file
    results = pd.concat(all_metrics)
    logging.info(results)
    results.to_csv(experiment_dir / "metrics.csv", index=False)
