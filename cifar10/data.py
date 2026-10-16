# data.py
import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_cifar10_loader(batch_size=64, num_workers=2, device="cpu", seed=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker,
        generator=g
    )
    return loader
