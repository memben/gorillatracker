import importlib

import torch
from torchvision.transforms import Compose, ToTensor

from gorillatracker.data_modules import QuadletDataModule, TripletDataModule


def get_dataset_class(pypath: str):
    parent = torch.utils.data.Dataset
    modpath, clsname = pypath.rsplit(".", 1)
    mod = importlib.import_module(modpath)
    cls = getattr(mod, clsname)
    assert issubclass(cls, parent), f"{cls} is not a subclass of {parent}"
    return cls


def _assert_tensor(x):
    assert isinstance(
        x, torch.Tensor
    ), f"GorillaTrackerDataset.get_transforms must contain ToTensor. Transformed result is {type(x)}"
    return x


def get_data_module(dataset_class_id, data_dir, batch_size, loss_mode: str, model_transforms):
    base = QuadletDataModule if loss_mode.startswith("online") else TripletDataModule
    dataset_class = get_dataset_class(dataset_class_id)
    transforms = Compose(
        [
            dataset_class.get_transforms() if hasattr(dataset_class, "get_transforms") else ToTensor(),
            _assert_tensor,
            model_transforms,
        ]
    )
    return base(data_dir, batch_size, dataset_class, transforms=transforms)