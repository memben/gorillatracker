from functools import partial

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Grayscale, ToTensor

from gorillatracker.data_modules import QuadletDataModule, TripletDataModule


def get_mnist_transforms():
    return Compose(
        [
            Grayscale(num_output_channels=3),
            ToTensor(),
        ]
    )


class MNISTTripletDataModule(TripletDataModule):
    def get_dataset_class(self):
        return partial(MNIST, download=True)

    def get_transforms(self):
        return get_mnist_transforms()


class MNISTQuadletDataModule(QuadletDataModule):
    def get_dataset_class(self):
        return partial(MNIST, download=True)

    def get_transforms(self):
        return get_mnist_transforms()


if __name__ == "__main__":
    mnist_dm = MNISTTripletDataModule("./mnist")
    mnist_dm.setup("test")
    for images, labels in mnist_dm.test_dataloader():
        print(labels)
        exit(1)

    mnist_dm = MNISTQuadletDataModule("./mnist")
    mnist_dm.setup("test")
    for v in mnist_dm.test_dataloader():
        print(v)
