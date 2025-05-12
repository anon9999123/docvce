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

                refined_cf_path = (
                    Path(str(file.parents[2]) + "-refined3")
                    / str(file.parents[1].name)
                    / file.name
                )
                if refined_cf_path.is_file():
                    shutil.copy(
                        refined_cf_path,
                        dest_subdir
                        / str(file.name.replace(".png", "") + "-refined3.png"),
                    )

                real_sample_path = (
                    Path(str(file.parents[2]))
                    / "real_samples/class_correct"
                    / file.name
                )
                if real_sample_path.is_file():
                    shutil.copy(
                        real_sample_path,
                        dest_subdir / str(file.name.replace(".png", "") + "-real.png"),
                    )


if __name__ == "__main__":
    for dataset in ["Tobacco3482", "RvlCdip", "ds4sd_DocLayNet"]:
        for model in [
            "convnext_base",
            "resnet50",
            "docvce.models.dit_model.DitModel",
        ]:
            src_directory = (
                f"output/counterfactual_latent_diffusion/{dataset}/basic_{model}/"
            )
            dest_directory = f"output/counterfactual_latent_diffusion/visualization_images/{dataset}/basic_{model}"
            print(f"Copying top 250 images from {src_directory} to {dest_directory}")
            copy_top_n_files(src_directory, dest_directory, n=250)
