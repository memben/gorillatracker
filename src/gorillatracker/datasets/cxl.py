from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset

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


if __name__ == "__main__":
    for img, label in CXLDataset(
        "data/splits/ground_truth-cxl-closedset--mintraincount-3-seed-42-train-70-val-15-test-15"
    ):
        print(img, label, img.filename)
