import os
from pathlib import Path

import hydra
import torch
import tqdm
from atria.core.task_runners.atria_data_processor import AtriaDataProcessor
from atria.core.utilities.pydantic_parser import atria_pydantic_parser
from torchvision.transforms.functional import to_pil_image


def save_image(image, path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def get_real_samples_split(real_samples_dir_path):
    real_samples_split = os.listdir(
        real_samples_dir_path / "real_samples" / "class_correct"
    )
    real_samples_split += os.listdir(
        real_samples_dir_path / "real_samples" / "class_incorrect"
    )
    return [x.split("_from")[0] for x in real_samples_split]


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="sfid_split_generator",
)
def app(cfg: AtriaDataProcessor) -> None:
    from hydra_zen import instantiate

    atria_data_processor: AtriaDataProcessor = instantiate(
        cfg, _convert_="object", _target_wrapper_=atria_pydantic_parser
    )
    atria_data_processor.init(hydra_config=None)

    real_samples_dir_path = Path(cfg.real_samples_dir_path)
    real_samples_split_1 = get_real_samples_split(real_samples_dir_path)

    if cfg.concat_test_and_train:
        test_dataset = torch.utils.data.ConcatDataset(
            [
                atria_data_processor._data_module.test_dataset,
                atria_data_processor._data_module.train_dataset,
            ]
        )
    else:
        test_dataset = atria_data_processor._data_module.test_dataset

    total_images_saved = 0
    for sample in tqdm.tqdm(
        test_dataset,
        "Saving 2nd split of real samples for sFID",
    ):
        if sample["__key__"] not in real_samples_split_1:
            image = to_pil_image(sample["image"] * 0.5 + 0.5)
            image_save_path = (
                real_samples_dir_path
                / "real_samples_split_2"
                / f"{sample['__key__']}.png"
            )
            save_image(image, image_save_path)
            total_images_saved += 1

        if total_images_saved == len(real_samples_split_1):
            break


if __name__ == "__main__":
    app()
