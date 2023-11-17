import os
import random
from typing import TYPE_CHECKING, Tuple

# from torchvision import transforms
import lightning as L
import torch
from PIL import Image
from print_on_steroids import logger

# from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import MNIST
from torchvision.transforms import RandAugment
from torchvision.transforms.functional import pad, resize, to_tensor

# import time

# import model


if TYPE_CHECKING:
    from train import TrainingArgs


def get_train_transforms():
    # TODO: add RandAugment # use magnitude 5 and 3 operations
    return RandAugment(num_ops=3, magnitude=9)


class MNISTDataset(Dataset):
    def __init__(self, mode="train", transform=get_train_transforms()) -> None:
        super().__init__()
        self.transform = transform
        self.mode = mode
        self.mnist = MNIST(root="data", train=True, download=True)
        self.data = sorted(self.mnist, key=lambda x: x[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # index is a tuple of 4 indices
        anchor_img, anchor_label = self.data[index[0]]
        positive_img, positive_label = self.data[index[1]]
        negative_img, negative_label = self.data[index[2]]
        negative_positive_img, negative_positive_label = self.data[index[3]]
        anchor_idx, positive_idx, negative_idx, negative_positive_idx = index

        assert (
            anchor_label == positive_label
            and anchor_label != negative_label
            and anchor_label != negative_positive_label
            and negative_label == negative_positive_label
        )

        # convert to 224x224x3
        # anchor_img = img_resize(anchor_img, size=32)
        # positive_img = img_resize(positive_img, size=32)
        # negative_img = img_resize(negative_img, size=32)

        if self.transform is not None and self.mode == "train":
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            negative_positive_img = self.transform(negative_positive_img)

        anchor_img = to_tensor(anchor_img)
        anchor_img = anchor_img.repeat(3, 1, 1)
        positive_img = to_tensor(positive_img)
        positive_img = positive_img.repeat(3, 1, 1)
        negative_img = to_tensor(negative_img)
        negative_img = negative_img.repeat(3, 1, 1)
        negative_positive_img = to_tensor(negative_positive_img)
        negative_positive_img = negative_positive_img.repeat(3, 1, 1)

        return (
            (anchor_img, anchor_label, anchor_idx),
            (positive_img, positive_label, positive_idx),
            (negative_img, negative_label, negative_idx),
            (negative_positive_img, negative_positive_label, negative_positive_idx),
        )


class MNISTSampler(Sampler):
    def __init__(self, mode="train"):
        super().__init__()
        self.mnist = MNIST(root="data", train=True, download=True)

        self.indices_with_labels = list(enumerate([label for _, label in sorted(self.mnist, key=lambda x: x[1])]))
        random.seed(42)  # NOTE: seed is set here to be able to reproduce results (bad practice - should be set in main)
        random.shuffle(self.indices_with_labels)

        if mode == "train":
            split_index = int(len(self.indices_with_labels) * 0.85)
            self.indices_with_labels = self.indices_with_labels[:split_index]
        elif mode == "val":
            split_index = int(len(self.indices_with_labels) * 0.85)
            self.indices_with_labels = self.indices_with_labels[split_index:]

        self.indices_with_labels.sort(key=lambda x: x[1])

        individuals = {}
        unique_labels = list(set(label for _, label in self.indices_with_labels))

        for i, label in enumerate(unique_labels):
            labels_count = sum(1 for _, lbl in self.indices_with_labels if lbl == label)
            start_index = sum(1 for _, lbl in self.indices_with_labels if lbl < label)
            individuals[label] = (start_index, labels_count)

        self.individuals = individuals

    def __iter__(self):
        indices_with_labels_copy = self.indices_with_labels.copy()
        random.shuffle(indices_with_labels_copy)

        for anchor_idx, anchor_label in indices_with_labels_copy:
            anchor_start, anchor_length = self.individuals[anchor_label]
            positive_idx = random.randint(anchor_start, anchor_start + anchor_length - 1)
            negative_idx = random.randint(0, len(self.indices_with_labels) - anchor_length - 1)
            negative_idx = negative_idx if negative_idx < anchor_start else negative_idx + anchor_length
            _, negative_label = self.indices_with_labels[negative_idx]
            negative_start, negative_length = self.individuals[negative_label]
            negative_positive_idx = random.randint(
                negative_start, negative_start + negative_length - 2
            )  # -2 because we need to exclude the negative_idx
            if negative_positive_idx >= negative_idx:
                negative_positive_idx += 1

            # print("Positive Label ", self.indices_with_labels[positive_idx][1])
            # print("Negative Label ", self.indices_with_labels[negative_idx][1])

            yield anchor_idx, self.indices_with_labels[positive_idx][0], self.indices_with_labels[negative_idx][
                0
            ], self.indices_with_labels[negative_positive_idx][0]

    def __len__(self):
        return len(self.indices_with_labels)


# Annahme: Daten liegen alle in einem Ordner (data_dir)
# Annahme: Bilder sind bereits auf Gesichter zugeschnitten
# Annahme: Name der Dateien entspricht Schema: <name_individuum>-<nr>-img-<nr>.jpg
class BristolDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        # Extract List of all individuals from data_dir corresponding to the
        self.data_dir = data_dir
        self.image_files = read_image_files(data_dir)
        # self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
        self.image_files.sort()

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: Tuple[int, int, int]):
        # load the images associated with the given indices
        anchor_idx, positive_idx, negative_idx = idx

        anchor_individual, anchor_img_name = self.image_files[anchor_idx].split("-")[0], "-".join(
            self.image_files[anchor_idx].split("-")[1:]
        )
        positive_individual, positive_img_name = self.image_files[positive_idx].split("-")[0], "-".join(
            self.image_files[positive_idx].split("-")[1:]
        )
        negative_individual, negative_img_name = self.image_files[negative_idx].split("-")[0], "-".join(
            self.image_files[negative_idx].split("-")[1:]
        )

        # Load image
        anchor_img = Image.open(os.path.join(self.data_dir, os.path.join(anchor_individual, anchor_img_name)))
        positive_img = Image.open(os.path.join(self.data_dir, os.path.join(positive_individual, positive_img_name)))
        negative_img = Image.open(os.path.join(self.data_dir, os.path.join(negative_individual, negative_img_name)))

        # Preprocess image and label as needed
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        anchor_img = img_resize(anchor_img, size=256)
        positive_img = img_resize(positive_img, size=256)
        negative_img = img_resize(negative_img, size=256)
        # convert to tensor
        anchor_img = to_tensor(anchor_img)
        positive_img = to_tensor(positive_img)
        negative_img = to_tensor(negative_img)

        # TODO: add labels to the returned stuff here
        return anchor_img, positive_img, negative_img


def img_resize(image, size):
    width, height = image.size
    if width > height:
        new_width = size
        new_height = int(height * size / width)
    else:
        new_height = size
        new_width = int(width * size / height)
    image = resize(image, (new_height, new_width))
    pad_width = size - new_width
    pad_height = size - new_height
    image = pad(image, (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))

    return image


# NOTE: beware of setting a seed to be able to reproduce results
class BristolSampler(torch.utils.data.Sampler):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        # self.batch_size = batch_size

        # image_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
        image_files = read_image_files(data_dir)
        image_files.sort()
        self.image_files = image_files

        individuals = list(set([f.split("-")[0] for f in image_files]))
        individuals_range_length = [len([f for f in self.image_files if f.startswith(i)]) for i in individuals]
        individuals_range_start = [sum(individuals_range_length[:i]) for i in range(len(individuals))]
        self.individuals = dict(zip(individuals, zip(individuals_range_start, individuals_range_length)))

    def __iter__(self):
        image_files_copy = self.image_files.copy()
        random.shuffle(image_files_copy)
        for anchor_idx, anchor_img_name in enumerate(image_files_copy):
            individual = anchor_img_name.split("-")[0]
            positive_idx = random.randint(
                self.individuals[individual][0], self.individuals[individual][0] + self.individuals[individual][1] - 1
            )
            negative_idx = random.randint(0, len(self.image_files) - self.individuals[individual][1] - 1)
            negative_idx = (
                negative_idx
                if negative_idx < self.individuals[individual][0]
                else negative_idx + self.individuals[individual][1]
            )

            yield anchor_idx, positive_idx, negative_idx

    def __len__(self):
        return len(self.image_files)


# read all directories in data_dir (a directory corresponds to one individual)
# returns a list of all image files in data_dir prefixed with the individual's name (and -)


def read_image_files(data_dir):
    individuals = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    image_files = []
    for individual in individuals:
        image_files.extend([individual + "-" + f for f in os.listdir(os.path.join(data_dir, individual))])

    return image_files


class GorillaDM(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        transform=get_train_transforms(),
    ):
        super().__init__()
        self.args = training_args
        self.train_dir = training_args.train_dir
        self.val_dir = training_args.val_dir
        self.test_dir = training_args.test_dir
        self.transform = transform

        logger.debug(f"Train data dir: {self.train_dir}")

        # self.local_rank = get_rank()

    # single gpu -> used for downloading and preprocessing which cannot be parallelized
    def prepare_data(self) -> None:
        pass  # nothing to do here

    def setup(self, stage):
        # TODO: implement train/val split
        # self.train_dataset = BristolDataset(data_dir=self.train_dir, transform=self.transform)
        # self.val_dataset = BristolDataset(data_dir=self.val_dir, transform=self.transform)
        # self.train_sampler = BristolSampler(data_dir=self.train_dir)
        # self.val_sampler = BristolSampler(data_dir=self.val_dir)

        self.train_dataset = MNISTDataset(mode="train")
        self.val_dataset = MNISTDataset(mode="val")
        self.train_sampler = MNISTSampler(mode="train")
        self.val_sampler = MNISTSampler(mode="val")

    def train_dataloader(self):
        pin_memory = True
        if self.args.accelerator != "cuda":
            pin_memory = False
        common_args = dict(
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=pin_memory,
        )
        return DataLoader(self.train_dataset, sampler=self.train_sampler, **common_args)

    def val_dataloader(self):
        pin_memory = True
        if self.args.accelerator != "cuda":
            pin_memory = False
        common_args = dict(
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=pin_memory,  # TODO: understand this # short answer: aims to optimize data transfer between the CPU and GPU
        )
        return DataLoader(self.val_dataset, sampler=self.val_sampler, **common_args)


if __name__ == "__main__":
    import model

    # set seed
    torch.manual_seed(42)
    random.seed(45)
    # test sampler and dataset with a single batch
    # data_dir = "/workspaces/gorillavision/datasets/face_detection/all_images_no_cropped_backup"
    # data_dir = "/workspaces/gorillavision/datasets/cxl/face_images_grouped"
    # sampler = BristolSampler(data_dir)
    # dataset = BristolDataset(data_dir=data_dir)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=4)
    # print(len(dataset))
    # batch = next(iter(dataloader))
    # # save an image
    # img = batch[0][0].permute(1, 2, 0)
    # Image.fromarray((img.numpy() * 255).astype("uint8")).save("test.jpg")

    model1 = model.EfficientNetV2Wrapper(
        model_name_or_path="",
        from_scratch=True,
        learning_rate=0.01,
        weight_decay=0,
        lr_schedule="lambda",
        warmup_epochs=1,
        lr_decay=0.99,
        lr_decay_interval=2,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        save_hyperparameters=True,
    )
    print(model1.model)

    # test dataloader
    # batch = next(iter(dataloader))
    # loss = model1._calculate_loss(batch)
    # # print(loss)

    # start_time = time.time()
    # for batch in dataloader:
    #     continue
    # end_time = time.time()
    # print(f"Time: {end_time - start_time} seconds")

    # test MNIST stuff
    sampler = MNISTSampler()
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=0)

    # test model output shape
    batch = next(iter(dataloader))
    out = model1(batch[0])
    out2 = model1.model(batch[0])
    print(out.shape)
    print(out2.shape)

    # print(len(dataset))
    # random.seed(45)
    # batch = next(iter(dataloader))
    # # save an image
    # anchor_img, positive_img, negative_img = batch
    # # anchor_img = anchor_img[0]
    # print(anchor_img.shape)
    # img = anchor_img[0]  # imgs are grayscale
    # img = img.permute(1, 2, 0)
    # img = img.squeeze()
    # Image.fromarray((img.numpy() * 255).astype("uint8")).save("test1.jpg")

    # img = positive_img[0]  # imgs are grayscale
    # img = img.permute(1, 2, 0)
    # img = img.squeeze()
    # Image.fromarray((img.numpy() * 255).astype("uint8")).save("test2.jpg")

    # img = negative_img[0]  # imgs are grayscale
    # img = img.permute(1, 2, 0)
    # img = img.squeeze()
    # Image.fromarray((img.numpy() * 255).astype("uint8")).save("test3.jpg")

    # # iterate over the dataloader to see if it works
    # start_time = time.time()
    # for batch in dataloader:
    #     continue
    # end_time = time.time()
    # print(f"Time: {end_time - start_time} seconds")
