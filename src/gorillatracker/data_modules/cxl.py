from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from gorillatracker.data_modules.data_modules import QuadletDataModule, TripletDataModule

Label = Union[int, str]


def get_samples(data_dir: Path) -> List[Tuple[Path, Label]]:
    """
    Assumed directory structure:
        data_dir/
            <label>_<...>.png

    """
    samples = []
    image_paths = data_dir.glob("**/*.png")
    for image_path in image_paths:
        label = image_path.name.split("_")[0]
        samples.append((image_path, label))
    return samples


class CXLDataset(Dataset):
    def __init__(self, data_dir, train: bool = True, transform=None):
        """
        Assumes directory structure:
            data_dir/
                train/
                    ...
                eval/
                    ...
        """
        dirpath = data_dir / Path("train" if train else "eval")
        self.samples = get_samples(dirpath)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.functional.pad(image, padding, 0, "constant")


def get_bristol_transforms():
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )


class CXLTripletDataModule(TripletDataModule):
    def get_dataset_class(self):
        return CXLDataset

    def get_transforms(self):
        return get_bristol_transforms()


class CXLQuadletDataModule(QuadletDataModule):
    def get_dataset_class(self):
        return CXLDataset

    def get_transforms(self):
        return get_bristol_transforms()


if __name__ == "__main__":
    # mnist_dm = CXLTripletDataModule(
    #     "data/splits/ground_truth-cxl-closedset--mintraincount-3-seed-42-train-70-val-15-test-15"
    # )
    # mnist_dm.setup("test")
    # print("setup test done")
    # for v in mnist_dm.test_dataloader():
    #     print(v)
    # print("run finished")

    mnist_dm = CXLQuadletDataModule(
        "data/splits/ground_truth-cxl-closedset--mintraincount-3-seed-42-train-70-val-15-test-15"
    )
    mnist_dm.setup("test")
    for v in mnist_dm.test_dataloader():
        print(v)
