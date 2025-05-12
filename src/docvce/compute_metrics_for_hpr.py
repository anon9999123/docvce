import argparse
import logging
import os
import warnings
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from docvce.models.dit_model import DitModel

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
        for img_file in tqdm(counterfactual_images_dir.glob("*.png")):
            self._counterfactual_image_paths.append(img_file)
            real_image_path = real_images_dir / "class_correct" / img_file.name
            if not real_image_path.exists():
                real_image_path = real_images_dir / "class_incorrect" / img_file.name
            self._real_image_paths.append(real_image_path)
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
        cf_target = int(
            self._counterfactual_image_paths[idx]
            .name.replace(".png", "")
            .split("to_")[1]
        )
        return real_img, counterfactual_img, cf_target

    def load_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        return self._transform(img)


class UnpairedCounterfactualDataset(CounterfactualDataset):
    def __init__(self, real_images_dir: str, counterfactual_images_dir: str):
        # set up the paths
        self.real_images_dir = Path(real_images_dir)
        self.counterfactual_images_dir = Path(counterfactual_images_dir)
        self._real_image_paths = []
        self._counterfactual_image_paths = []
        for img_file in tqdm(counterfactual_images_dir.glob("*.png")):
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
    parser.add_argument(
        "--model_type",
        required="convnext",
        type=str,
        help="Type of the model to compute the confidence scores",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Path to the model checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()

    # setup model
    if args.dataset == "rvlcdip":
        num_classes = 16
    elif args.dataset == "tobacco3482":
        num_classes = 10
    elif args.dataset == "doclaynet":
        num_classes = 6
    else:
        raise ValueError("Dataset not supported")

    if args.model_type == "convnext_base":
        from timm import create_model

        model = create_model("convnext_base", pretrained=False, num_classes=num_classes)
        checkpoint = torch.load(args.model_path)
        checkpoint = {
            k.replace("model.", ""): v
            for k, v in checkpoint["task_module"]["model"].items()
        }
        model.load_state_dict(checkpoint, strict=True)
    elif args.model_type == "resnet50":
        from timm import create_model

        model = create_model("resnet50", pretrained=False, num_classes=num_classes)
        checkpoint = torch.load(args.model_path)
        checkpoint = {
            k.replace("model.", ""): v
            for k, v in checkpoint["task_module"]["model"].items()
        }
        model.load_state_dict(checkpoint, strict=True)
    elif args.model_type == "dit_b":
        checkpoint = torch.load(args.model_path)
        checkpoint = {k: v for k, v in checkpoint["task_module"]["model"].items()}
        model = DitModel(num_labels=num_classes)
        model.load_state_dict(checkpoint, strict=True)
    else:
        raise ValueError("Model type not supported")
    model.eval()
    model = model.to(0)

    experiment_dir = Path(args.experiment_dir)
    if "-refined" in experiment_dir.name:
        real_samples_split_1_dir_path = (
            experiment_dir.parent
            / experiment_dir.name.replace("-refined3", "")
            / "real_samples"
        )
        real_samples_split_2_dir_path = (
            experiment_dir.parent
            / experiment_dir.name.replace("-refined3", "")
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
            if (run_dir / "metrics.csv").exists():
                all_metrics.append(pd.read_csv(run_dir / "metrics.csv"))
                continue
            results = []
            logging.info(f"Run dir: {run_dir}")
            dataset_real_samples_split_1 = CounterfactualDataset(
                real_samples_split_1_dir_path, run_dir
            )
            dataset_real_samples_split_2 = UnpairedCounterfactualDataset(
                real_samples_split_2_dir_path, run_dir
            )
            results.append(
                {
                    "run_dir": run_dir.name,
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
                batch_size=64,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                pin_memory=True,
            )

            logging.info("Computing closeness metrics")
            for real, cf, cf_target in tqdm(dataloader_split_1):
                cf_target = cf_target.to(0)
                real = real.to(0, dtype=torch.float)
                cf = cf.to(0, dtype=torch.float)
                bsz = real.shape[0]
                diff = real.view(bsz, -1) - cf.view(bsz, -1)
                l1_norm_sum = torch.norm(diff, p=1, dim=-1)
                l2_norm_sum = torch.norm(diff, p=2, dim=-1)
                l1_norm_mean = l1_norm_sum / diff.shape[-1]
                l2_norm_mean = l2_norm_sum / diff.shape[-1]
                lpips_loss = lpips_model(real, cf, normalize=True)
                with torch.no_grad():
                    cf_transformed = (cf - 0.5) / 0.5
                    counterfactual_logits = model(cf_transformed)
                    counterfactual_probs = torch.softmax(counterfactual_logits, dim=1)
                    conf_scores = counterfactual_probs[
                        range(counterfactual_probs.size(0)), cf_target.view(-1)
                    ]
                    predicted_label = counterfactual_logits.argmax(-1)
                    flipped = (predicted_label == cf_target).float()

                for i in range(bsz):
                    results.append(
                        {
                            "run_dir": run_dir.name,
                            "l1_norm_sum": l1_norm_sum[i].item(),
                            "l2_norm_sum": l2_norm_sum[i].item(),
                            "l1_norm_mean": l1_norm_mean[i].item(),
                            "l2_norm_mean": l2_norm_mean[i].item(),
                            "lpips_loss": lpips_loss[i].item(),
                            "flipped": flipped[i].item(),
                            "conf_score": conf_scores[i].item(),
                        }
                    )

            logging.info("Computing FID")
            fid = compute_fid(dataloader_split_1)
            dataloader_split_2 = data.DataLoader(
                dataset_real_samples_split_2,
                batch_size=64,
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
