import numpy as np
import torchvision.transforms as transforms

from gorillatracker.data_modules.data_modules import QuadletDataModule, TripletDataModule
from gorillatracker.datasets.bristol import BristolDataset


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
            transforms.Resize(124),
            transforms.ToTensor(),
        ]
    )


class BristolTripletDataModule(TripletDataModule):
    def get_dataset_class(self):
        return BristolDataset

    def get_transforms(self):
        return get_bristol_transforms()


class BristolQuadletDataModule(QuadletDataModule):
    def get_dataset_class(self):
        return BristolDataset

    def get_transforms(self):
        return get_bristol_transforms()


if __name__ == "__main__":
    mnist_dm = BristolTripletDataModule("datasets/cxl-bristol_0_75")
    mnist_dm.setup("test")
    print("setup test done")
    for v in mnist_dm.test_dataloader():
        print(v)
    print("run finished")

    mnist_dm = BristolQuadletDataModule("datasets/cxl-bristol_0_75")
    mnist_dm.setup("test")
    for v in mnist_dm.test_dataloader():
        print(v)
