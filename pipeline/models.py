import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply SN after all layers are created.

    This preserves identical underlying weight initialization for SN and non-SN
    variants when they share a seed; SN's auxiliary vectors are added only after
    the raw network parameters already exist.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            setattr(module, name, spectral_norm(child))
        else:
            apply_spectral_norm(child)
    return module


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim: int = 128, image_size: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.image_size = image_size

        if image_size == 32:
            start_ch = 128
            self.fc = nn.Linear(z_dim, start_ch * 4 * 4)
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
                nn.Tanh(),
            )
        elif image_size == 64:
            start_ch = 256
            self.fc = nn.Linear(z_dim, start_ch * 4 * 4)
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
                nn.Tanh(),
            )
        else:
            raise ValueError("DCGAN currently supports image_size 32 or 64")

        self.start_ch = start_ch

    def forward(self, z):
        x = self.fc(z).view(-1, self.start_ch, 4, 4)
        return self.conv(x)


class DCGANDiscriminator(nn.Module):
    def __init__(self, image_size: int = 32, use_sn: bool = False):
        super().__init__()
        self.image_size = image_size

        if image_size == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Linear(256 * 4 * 4, 1)
        elif image_size == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 512, 3, 2, 1),
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Linear(512 * 4 * 4, 1)
        else:
            raise ValueError("DCGAN currently supports image_size 32 or 64")

        if use_sn:
            apply_spectral_norm(self)

    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc(h)


class ResGenBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.relu(self.bn1(x), inplace=False)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=False)
        h = self.conv2(h)

        s = F.interpolate(x, scale_factor=2, mode="nearest")
        s = self.shortcut(s)
        return h + s


class ResNetGenerator32(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.b1 = ResGenBlock(256, 256)  # 4 -> 8
        self.b2 = ResGenBlock(256, 128)  # 8 -> 16
        self.b3 = ResGenBlock(128, 64)   # 16 -> 32
        self.bn = nn.BatchNorm2d(64)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = F.relu(self.bn(h), inplace=False)
        return torch.tanh(self.out(h))


class ResDiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first: bool = False):
        super().__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = x if self.first else F.relu(x, inplace=False)
        h = self.conv1(h)
        h = F.relu(h, inplace=False)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)

        s = self.shortcut(x)
        s = F.avg_pool2d(s, 2)
        return h + s


class ResNetDiscriminator32(nn.Module):
    def __init__(self, use_sn: bool = False):
        super().__init__()
        self.b1 = ResDiscBlock(3, 64, first=True)   # 32 -> 16
        self.b2 = ResDiscBlock(64, 128)             # 16 -> 8
        self.b3 = ResDiscBlock(128, 256)            # 8 -> 4
        self.fc = nn.Linear(256, 1)
        if use_sn:
            apply_spectral_norm(self)

    def forward(self, x):
        h = self.b1(x)
        h = self.b2(h)
        h = self.b3(h)
        h = F.relu(h, inplace=False)
        h = h.sum(dim=(2, 3))
        return self.fc(h)


def build_generator(architecture: str, image_size: int, z_dim: int):
    architecture = architecture.lower()
    if architecture == "dcgan":
        return DCGANGenerator(z_dim=z_dim, image_size=image_size)
    if architecture == "resnet":
        if image_size != 32:
            raise ValueError("The ResNet validation architecture currently supports 32x32 only")
        return ResNetGenerator32(z_dim=z_dim)
    raise ValueError(f"Unknown architecture: {architecture}")


def build_discriminator(architecture: str, image_size: int, use_sn: bool):
    architecture = architecture.lower()
    if architecture == "dcgan":
        return DCGANDiscriminator(image_size=image_size, use_sn=use_sn)
    if architecture == "resnet":
        if image_size != 32:
            raise ValueError("The ResNet validation architecture currently supports 32x32 only")
        return ResNetDiscriminator32(use_sn=use_sn)
    raise ValueError(f"Unknown architecture: {architecture}")
