import lightning as L
from print_on_steroids import logger
import torch
from torch.optim import Adam
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import importlib

eps = 1e-16  # an arbitrary small value to be used for numerical stability tricks


def get_triplet_mask(labels):
    """compute a mask for valid triplets
    Args:
    labels: Batch of integer labels. shape: (batch_size,)
    Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
    """
    # step 1 - get a mask for distinct indices

    # shape: (batch_size, batch_size)
    batch_size = labels.size()[0]
    indices_equal = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)
    # shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2).repeat(1, 1, batch_size)
    # shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1).repeat(1, batch_size, 1)
    # shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0).repeat(batch_size, 1, 1)
    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(
        torch.logical_and(i_not_equal_j, i_not_equal_k),
        j_not_equal_k,
    )

    # step 2 - get a mask for valid anchor-positive-negative triplets

    # shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    # shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2).repeat(1, 1, batch_size)
    # shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1).repeat(1, batch_size, 1)
    # shape: (batch_size, batch_size, batch_size)
    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # step 3 - combine two masks
    mask = torch.logical_and(distinct_indices, valid_indices)

    return mask


def euclidean_distance_matrix(x):
    """Efficient computation of Euclidean distance matrix

    Args:
    x: Input tensor of shape (batch_size, embedding_dim)

    Returns:
    Distance matrix of shape (batch_size, batch_size)
    """
    # step 1 - compute the dot product

    # shape: (batch_size, batch_size)
    dot_product = torch.mm(x, x.t())

    # step 2 - extract the squared Euclidean norm from the diagonal

    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)

    # step 3 - compute squared Euclidean distances

    # shape: (batch_size, batch_size)
    distance_matrix = F.relu(squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1))

    # step 4 - compute the non-squared distances

    # handle numerical stability
    # derivative of the square root operation applied to 0 is infinite
    # we need to handle by setting any 0 to eps
    mask = (distance_matrix == 0.0).float()

    # use this mask to set indices with a value of 0 to eps
    distance_matrix_stable = torch.sqrt(distance_matrix + mask * eps) * (1.0 - mask)

    return distance_matrix_stable


# TODO: add option hard for only using the max distance to a positive and min to negative per anchor
class TripletLossOnline(nn.Module):
    """Uses all valid triplets to compute Triplet loss

    Args:
      margin: Margin value in the Triplet Loss equation
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """computes loss value.

        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)

        Returns:
          Scalar loss value.
        """
        # step 1 - get distance matrix
        # shape: (batch_size, batch_size)
        distance_matrix = euclidean_distance_matrix(embeddings)

        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

        # shape: (batch_size, batch_size, 1)
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        # shape: (batch_size, 1, batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin

        # step 3 - filter out invalid or easy triplets by setting their loss values to 0

        # shape: (batch_size, batch_size, batch_size)
        mask = get_triplet_mask(labels)
        triplet_loss *= mask
        # easy triplets have negative loss values
        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (mask.float() > eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

        return triplet_loss


class TripletLossOffline(nn.Module):
    """Compute Triplet loss for a batch of generated triplets

    Args:
      margin: Margin value in the Triplet Loss equation
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """computes loss value.

        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)

        Returns:
          Scalar loss value.
        """
        # reconstruct triplets from embeddings and labels
        triplet_base_amount = embeddings.size()[0] // 3
        anchor_embeddings = embeddings[:triplet_base_amount]
        positive_embeddings = embeddings[triplet_base_amount : 2 * triplet_base_amount]
        negative_embeddings = embeddings[2 * triplet_base_amount :]

        distance_positive = torch.functional.norm(anchor_embeddings - positive_embeddings, dim=1)
        distance_negative = torch.functional.norm(anchor_embeddings - negative_embeddings, dim=1)

        losses = torch.relu(distance_positive - distance_negative + self.margin).mean()
        return losses.mean()  # , distance_positive.mean(), distance_negative.mean()


class BaseModule(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        # model_kwargs: dict,
        from_scratch: bool,  # TODO
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

        # Remove LR scheduling for now
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.margin = margin

        ##### Load the model here #####
        # TODO
        self.model = None
        self.from_scratch = from_scratch

        ##### Create Table embeddings_table
        self.embeddings_table_columns = ["label", "embedding"]
        self.embeddings_table = pd.DataFrame(columns=self.embeddings_table_columns)
        self.embedding_size = embedding_size

        # TODO
        self.triplet_loss = TripletLossOnline(margin=margin)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        anchor, positive, negative, negative_positive = batch  # format is -> Tupel[Tupel(int,2), 3]

        anchor, anchor_label, anchor_idx = anchor
        positive, positive_label, positive_idx = positive
        negative, negative_label, negative_idx = negative
        negative_positive, negative_positive_label, negative_positive_idx = negative_positive

        images = torch.cat((anchor, positive, negative, negative_positive), dim=0)
        # labels = torch.cat((anchor_idx, positive_idx, negative_idx), dim=0)
        labels = torch.cat((anchor_label, positive_label, negative_label, negative_positive_label), dim=0)
        embeddings = self(images)

        loss = self.triplet_loss(embeddings, labels)
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=True)

        return loss

        # self.log("train_distance_positive", distance_positive, on_step=True, prog_bar=True, sync_dist=True)
        # self.log("train_distance_negative", distance_negative, on_step=True, prog_bar=True, sync_dist=True)

    def add_validation_embeddings(self, batch_embeddings, batch_labels):
        # feed the anchor, positive, negative images to the model

        # log the embedded anchor to x (create dataframe for current batch -> then concatenate df with self.embeddings_table)
        embeddings = torch.reshape(
            batch_embeddings, (-1, self.embedding_size)
        )  # get tensor of shape (batch_size, embedding_size)
        embeddings = embeddings.cpu()

        assert len(self.embeddings_table_columns) == 2
        data = {
            self.embeddings_table_columns[0]: batch_labels.tolist(),
            self.embeddings_table_columns[1]: [embedding.numpy() for embedding in embeddings],
        }

        data = pd.DataFrame(data)
        self.embeddings_table = pd.concat([data, self.embeddings_table], ignore_index=True)

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative, negative_positive = batch  # format is -> Tupel[Tupel(int,2), 3]

        anchor, anchor_label, anchor_idx = anchor
        positive, positive_label, positive_idx = positive
        negative, negative_label, negative_idx = negative
        negative_positive, negative_positive_label, negative_positive_idx = negative_positive

        images = torch.cat((anchor, positive, negative, negative_positive), dim=0)
        # labels = torch.cat((anchor_idx, positive_idx, negative_idx), dim=0)
        labels = torch.cat((anchor_label, positive_label, negative_label, negative_positive_label), dim=0)
        loss_embeddings = self(images)

        self.add_validation_embeddings(loss_embeddings[: len(anchor)], anchor_label)

        loss = self.triplet_loss(loss_embeddings, labels)

        self.log("val_loss", loss, on_step=True, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.learning_rate}, weight decay: {self.weight_decay} and warmup epochs: {self.warmup_epochs}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        optimized_named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in optimized_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in optimized_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adam(
            optimizer_parameters,
            self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,  # You can also tune this
        )

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs * self.learning_rate
            else:
                num_decay_cycles = (epoch - self.warmup_epochs) // self.lr_decay_interval
                return (self.lr_decay**num_decay_cycles) * self.learning_rate

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": self.lr_decay_interval},
        }


class EfficientNetV2Wrapper(BaseModule):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = (
            efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            if kwargs.get("from_scratch", False)
            else efficientnet_v2_l()
        )

        self.model.classifier = nn.Linear(1280, self.embedding_size)


custom_model_cls = {"EfficientNetV2_Large": EfficientNetV2Wrapper}


def get_model_cls(model_name: str):
    model_cls = custom_model_cls.get(model_name, None)
    if not model_cls:
        module, cls = model_name.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module), cls)
    return model_cls
