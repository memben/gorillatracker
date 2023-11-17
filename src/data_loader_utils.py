import itertools
from collections import defaultdict
from typing import Any, List, Tuple

import torch
from torch.utils.data.dataloader import DataLoader, Sampler


def generate_labelsection(sorted_value_labels: List[Tuple[Any, Any]]):
    n = len(sorted_value_labels)
    labelsection = defaultdict(lambda: (None, None))
    prev_label = None
    prev_start = None
    for i, (_, label) in enumerate(sorted_value_labels):
        assert prev_label is None or prev_label <= label, "dataset passed to TripletSampler must be label-sorted."
        if prev_label is None:
            prev_label = label
            prev_start = i
        elif prev_label != label:
            labelsection[prev_label] = (prev_start, i - prev_start)
            prev_label = label
            prev_start = i
    if prev_label:
        labelsection[prev_label] = (prev_start, n - prev_start)
    return labelsection


def iter_index_permutations_generator(n: int):
    """
    returns all possibel index permutations in a systematic order.
    WARN(liamvdv): the systematic order makes the first indices occure predominantly.
                   This screws the batch sampling towards lower labels (since dataset
                   is passed sorted).
    Behaviour Example:
        (0, 1, 2)
        (0, 2, 1)
        (1, 0, 2)
        (1, 2, 0)
        (2, 0, 1)
        (2, 1, 0)
    """
    return itertools.permutations(list(range(n)))


def index_permuation_generator(n: int):
    """
    returns a completly random index permutation. Does not ensure all permuations are seen.
    Reproducable via torch.seed.
    """
    while True:
        yield torch.randperm(n).tolist()


def randint_except(start: int, end: int, excluded: int) -> int:
    """[start, end)"""
    size = end - start
    if size == 0:
        raise ValueError("class must have at least 1 instance")
    elif size == 1:
        # TODO(liamvdv): filter out 1 instances?
        # raise ValueError("class must have >=2 instances")
        return start
    while True:
        idx = torch.randint(start, end, (1,)).item()
        if idx != excluded:
            return idx


class ToTripletDataset:
    def __init__(self, dataset, transform=lambda x: x):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idxs):
        if torch.is_tensor(idxs):
            idxs = idxs.tolist()
        a, p, n = idxs
        xa, ya = self.dataset[a]
        xa = self.transform(xa)
        xp, yp = self.dataset[p]
        xp = self.transform(xp)
        xn, yn = self.dataset[n]
        xn = self.transform(xn)
        return (xa, xp, xn), (ya, yp, yn)


class TripletSampler(Sampler):
    """
    dataset must not be shuffled
    """

    # dataset must be label sorted!
    def __init__(self, sorted_dataset, shuffled_indices_generator=index_permuation_generator):
        self.dataset = sorted_dataset
        self.n = len(self.dataset)
        self.labelsection = generate_labelsection(self.dataset)
        self.shuffled_indices_generator = shuffled_indices_generator(self.n)

    def any_sample_not(self, label) -> int:
        start, length = self.labelsection[label]
        end = start + length
        i = torch.randint(self.n - length, (1,)).item()
        if start <= i and i < end:
            i += length
        return i

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        anchor_shuffle = next(self.shuffled_indices_generator)
        for anchor in anchor_shuffle:
            anchor_label = self.dataset[anchor][1]
            astart, alength = self.labelsection[anchor_label]
            positive = randint_except(astart, astart + alength, anchor)
            negative = self.any_sample_not(anchor_label)
            yield anchor, positive, negative


def TripletDataLoader(dataset, batch_size):
    """
    TripletDataLoader will take any Dataset that returns a single sample in the form of 
    (value, label) on __getitem__ and transform it into an efficient Triplet DataLoader.
    """
    label_sorted_dataset = sorted(dataset, key=lambda t: t[1])
    sampler = TripletSampler(label_sorted_dataset)
    final_dataset = ToTripletDataset(label_sorted_dataset)
    return DataLoader(final_dataset, sampler=sampler, shuffle=False, batch_size=batch_size)
