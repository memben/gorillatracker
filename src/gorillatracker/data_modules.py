import logging

import lightning as L
from torch.utils.data import random_split

from gorillatracker.data_loaders import QuadletDataLoader, TripletDataLoader

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, dataset_class=None, transforms=None):
        super().__init__()
        self.transforms = transforms or self.get_transforms()
        self.dataset_class = dataset_class or self.get_dataset_class()
        self.data_dir = data_dir
        self.batch_size = batch_size

    @classmethod
    def from_training_args(cls, args):
        return cls(data_dir=args.data_dir, batch_size=args.batch_size)

    def get_transforms(self):
        return None

    def get_dataset_class(self):
        raise Exception("must provide a dataset_cls argument or subclass")

    # TODO(liamvdv): what is `stage` for?
    def setup(self, stage: str):
        logger.info(f"setup {stage} for {type(self.dataset_class)}")
        if stage == "test":
            self.test = self.dataset_class(self.data_dir, train=False, transform=self.transforms)
        elif stage == "predict":
            self.predict = self.dataset_class(self.data_dir, train=False, transform=self.transforms)
        elif stage == "fit" or stage == "validate":
            full = self.dataset_class(self.data_dir, train=True, transform=self.transforms)
            self.train, self.val = random_split(full, [0.8, 0.2])  # TODO(liamvdv): deterministic?
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def train_dataloader(self):
        return TripletDataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return TripletDataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return TripletDataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return TripletDataLoader(self.predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # NOTE(liamvdv): used to clean-up when the run is finished
        pass


class QuadletDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, dataset_class=None, transforms=None):
        super().__init__()
        self.transforms = transforms or self.get_transforms()
        self.dataset_class = dataset_class or self.get_dataset_class()
        self.data_dir = data_dir
        self.batch_size = batch_size

    @classmethod
    def from_training_args(cls, args):
        return cls(data_dir=args.data_dir, batch_size=args.batch_size)

    def get_transforms(self):
        return None

    def get_dataset_class(self):
        raise Exception("must provide a dataset_cls argument or subclass")

    # TODO(liamvdv): what is `stage` for?
    def setup(self, stage: str):
        logger.info(f"setup {stage} for {type(self.dataset_class)}")
        if stage == "test":
            self.test = self.dataset_class(self.data_dir, train=False, transform=self.transforms)
        elif stage == "predict":
            self.predict = self.dataset_class(self.data_dir, train=False, transform=self.transforms)
        elif stage == "fit" or stage == "validate":
            full = self.dataset_class(self.data_dir, train=True, transform=self.transforms)
            self.train, self.val = random_split(full, [0.8, 0.2])  # TODO(liamvdv): deterministic?
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def train_dataloader(self):
        return QuadletDataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return QuadletDataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return QuadletDataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return QuadletDataLoader(self.predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # NOTE(liamvdv): used to clean-up when the run is finished
        pass
