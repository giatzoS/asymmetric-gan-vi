import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import datasets as tv_datasets, transforms


def seed_worker(worker_id: int):
    """Seed NumPy/Python inside each DataLoader worker reproducibly."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _train_transform(image_size: int):
    ops = []
    if image_size != 32:
        ops.extend([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(ops)


def _eval_transform(image_size: int):
    ops = []
    if image_size != 32:
        ops.extend([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class PlantVillageOfficialSplit(Dataset):
    """PlantVillage RGB using the official leaf-group-aware split files."""

    def __init__(
        self,
        split: str,
        image_size: int,
        data_root: Optional[str],
        train: bool,
    ):
        if split not in {"train", "test"}:
            raise ValueError("PlantVillage split must be 'train' or 'test'")

        root = Path(data_root or "./data/plantvillage")

        self.image_root = root / "extracted"
        split_file = root / "splits" / f"color_{split}.txt"

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"PlantVillage extracted directory not found: {self.image_root}"
            )

        if not split_file.exists():
            raise FileNotFoundError(
                f"PlantVillage split file not found: {split_file}"
            )

        relative_paths = [
            line.strip()
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]

        self.paths = [self.image_root / rel for rel in relative_paths]

        if not self.paths:
            raise RuntimeError(f"No images found in split file: {split_file}")

        if not self.paths[0].exists():
            raise FileNotFoundError(
                f"First PlantVillage image does not exist: {self.paths[0]}"
            )

        self.transform = (
            _train_transform(image_size)
            if train
            else _eval_transform(image_size)
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[int(idx)]

        with Image.open(path) as img:
            image = img.convert("RGB")

        x = self.transform(image)

        # Labels are not used for unconditional GAN training.
        return x, 0


def build_train_dataset(
    dataset_name: str,
    data_root: str,
    image_size: int,
    split_seed: int = 123,
    train_fraction: float = 0.80,
):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        return tv_datasets.CIFAR10(
            root=data_root or "./data",
            train=True,
            download=True,
            transform=_train_transform(32),
        )

    if dataset_name == "plantvillage":
        # split_seed/train_fraction are intentionally ignored: PlantVillage uses
        # its published leaf-group-aware 80/20 train/test split.
        return PlantVillageOfficialSplit(
            split="train",
            image_size=image_size,
            data_root=data_root or "./data/plantvillage",
            train=True,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_eval_dataset(
    dataset_name: str,
    data_root: str,
    image_size: int,
    split_seed: int = 123,
    train_fraction: float = 0.80,
):
    """Return an unnormalized [0,1] dataset for the real FID/KID reference."""
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        return tv_datasets.CIFAR10(
            root=data_root or "./data",
            train=True,
            download=True,
            transform=_eval_transform(32),
        )

    if dataset_name == "plantvillage":
        return PlantVillageOfficialSplit(
            split="test",
            image_size=image_size,
            data_root=data_root or "./data/plantvillage",
            train=False,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_train_loader(
    dataset_name: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    device: str,
    seed: int,
    num_workers: int = 2,
    split_seed: int = 123,
    train_fraction: float = 0.80,
):
    """Create a fresh loader for one method/run.

    A fresh loader is intentional: methods sharing the same seed receive the
    same shuffle sequence instead of inheriting an advanced DataLoader RNG state.
    """
    dataset = build_train_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
        split_seed=split_seed,
        train_fraction=train_fraction,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=_loader_generator(seed),
        persistent_workers=(num_workers > 0),
    )
