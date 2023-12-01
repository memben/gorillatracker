import importlib

import lightning as L
import pandas as pd
import timm
import torch
import torchvision.transforms as transforms
from print_on_steroids import logger
from torch.optim import AdamW
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l

from gorillatracker.triplet_loss import get_triplet_loss


class BaseModule(L.LightningModule):
    """
    must be subclassed and set self.model = ...
    """

    def __init__(
        self,
        model_name_or_path: str,
        # model_kwargs: dict,
        from_scratch: bool,  # TODO
        loss_mode: str,
        learning_rate: float,
        weight_decay: float,
        lr_schedule: str,
        warmup_epochs: int,
        lr_decay: float,
        lr_decay_interval: int,
        beta1: float,
        beta2: float,
        epsilon: float = 1e-8,
        save_hyperparameters: bool = True,
        margin: float = 0.5,
        embedding_size: int = 256,
    ) -> None:
        super().__init__()

        if save_hyperparameters:
            self.save_hyperparameters(ignore=["save_hyperparameters"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # TODO(all): a working learning rate scheduler has to be implemented (
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval
        # )

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.margin = margin

        self.model = None
        self.from_scratch = from_scratch
        self.embedding_size = embedding_size

        ##### Create Table embeddings_table
        self.embeddings_table_columns = ["label", "embedding"]
        self.embeddings_table = pd.DataFrame(columns=self.embeddings_table_columns)

        # TODO(rob2u): rename loss mode
        self.triplet_loss = get_triplet_loss(loss_mode, margin)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch  # embeddings either (ap, a, an, n) oder (a, p, n)
        vec = torch.cat(images, dim=0)
        embeddings = self.forward(vec)
        labels = (
            torch.cat(labels, dim=0) if torch.is_tensor(labels[0]) else [label for group in labels for label in group]
        )
        loss, pos_dist, neg_dist = self.triplet_loss(embeddings, labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/positive_distance", pos_dist, on_step=True)
        self.log("train/negative_distance", neg_dist, on_step=True)
        return loss

    def add_validation_embeddings(self, anchor_embeddings, anchor_labels):
        # save anchor embeddings of validation step for later analysis in W&B
        embeddings = torch.reshape(anchor_embeddings, (-1, self.embedding_size))
        embeddings = embeddings.cpu()

        assert len(self.embeddings_table_columns) == 2
        data = {
            self.embeddings_table_columns[0]: anchor_labels.tolist()
            if torch.is_tensor(anchor_labels)
            else anchor_labels,
            self.embeddings_table_columns[1]: [embedding.numpy() for embedding in embeddings],
        }

        data = pd.DataFrame(data)
        self.embeddings_table = pd.concat([data, self.embeddings_table], ignore_index=True)
        # NOTE(rob2u): will get flushed by W&B Callback on val epoch end.

    def validation_step(self, batch, batch_idx):
        images, labels = batch  # embeddings either (ap, a, an, n) oder (a, p, n)
        n_achors = len(images[0])
        vec = torch.cat(images, dim=0)
        labels = (
            torch.cat(labels, dim=0) if torch.is_tensor(labels[0]) else [label for group in labels for label in group]
        )
        embeddings = self.forward(vec)

        self.add_validation_embeddings(embeddings[:n_achors], labels[:n_achors])
        loss, pos_dist, neg_dist = self.triplet_loss(embeddings, labels)
        self.log("val/loss", loss, on_step=True, sync_dist=True, prog_bar=True)
        self.log("val/positive_distance", pos_dist, on_step=True)
        self.log("val/negative_distance", neg_dist, on_step=True)
        return loss

    def configure_optimizers(self):
        # TODO(all): add lr_scheduler based on
        #            self.lr_schedule, self.warmup_epochs, self.lr_decay,
        #            self.lr_decay_interval.

        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup epochs: {self.warmup_epochs}"
            )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay,
        )
        return {"optimizer": optimizer}

    @staticmethod
    def get_tensor_transforms():
        raise NotImplementedError(
            "Please implement this method in your subclass: resizes, normalizations, etc. To apply nothing, return the identity function `lambda x: x`"
        )


class EfficientNetV2Wrapper(BaseModule):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        is_from_scratch = kwargs.get("from_scratch", False)
        self.model = (
            efficientnet_v2_l()
            if is_from_scratch
            else efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        )
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=self.embedding_size),
        )

    def forward(self, x):
        return self.model(x)

    @classmethod
    def get_tensor_transforms(cls):
        # NOTE(liamvdv): Efficient net can handle multiple image sizes. Thus we
        #                don't specify it here. Be aware.
        #                You would usually use
        #                transforms.Resize((224, 224), antialias=True)
        #                but for e. g. MNIST this will drop batch sizes from
        #                512 to 8.
        return lambda x: x


class SwinV2BaseWrapper(BaseModule):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        swin_model = "swinv2_base_window12_192.ms_in22k"
        self.model = (
            timm.create_model(swin_model, pretrained=False)
            if kwargs.get("from_scratch", False)
            else timm.create_model(swin_model, pretrained=True)
        )
        self.model.head.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embedding_size),
        )

    def forward(self, x):
        return self.model(x)

    @classmethod
    def get_tensor_transforms(cls):
        return transforms.Resize((192), antialias=True)


# NOTE(liamvdv): Register custom model backbones here.
custom_model_cls = {"EfficientNetV2_Large": EfficientNetV2Wrapper, "SwinV2Base": SwinV2BaseWrapper}


def get_model_cls(model_name: str):
    model_cls = custom_model_cls.get(model_name, None)
    if not model_cls:
        module, cls = model_name.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module), cls)
    return model_cls