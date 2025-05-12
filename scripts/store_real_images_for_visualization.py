import os
import shutil
from pathlib import Path


def copy_top_n_files(src_dir, dest_dir, n):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    if not src_path.is_dir():
        raise ValueError(
            f"Source directory {src_dir} does not exist or is not a directory"
        )

    for subdir in src_path.iterdir():
        if subdir.is_dir():
            if "refined" in subdir.name or "real" in subdir.name:
                continue
            files = sorted(
                (
                    (subdir / "class_correct_cf_correct").iterdir()
                    if "refined" not in src_path.name
                    else subdir.iterdir()
                ),
                key=os.path.getmtime,
                reverse=True,
            )
            files = [file for file in files if file.suffix == ".png"][:n]
            dest_subdir = dest_path / subdir.name
            dest_subdir.mkdir(parents=True, exist_ok=True)

            for file in files:
                if file.is_file():
                    shutil.copy(file, dest_subdir / file.name)


if __name__ == "__main__":
    for dataset in ["Tobacco3482", "RvlCdip", "ds4sd_DocLayNet"]:
        src_directory = (
            f"output/counterfactual_latent_diffusion/convnext_base/real_samples"
        )
        dest_directory = f"output/counterfactual_latent_diffusion/visualization_images/convnext_base/real_samples"
        copy_top_n_files(src_directory, dest_directory)
