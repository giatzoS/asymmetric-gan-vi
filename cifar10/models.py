# models.py
import torch.nn as nn
from torch.nn.utils import spectral_norm

def maybe_sn(layer, use_sn):
    return spectral_norm(layer) if use_sn else layer

class Generator(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128 * 4 * 4)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, use_sn=True):
        super().__init__()
        self.conv = nn.Sequential(
            maybe_sn(nn.Conv2d(3, 64, 3, 2, 1), use_sn),
            nn.LeakyReLU(0.2),

            maybe_sn(nn.Conv2d(64, 128, 3, 2, 1), use_sn),
            nn.LeakyReLU(0.2),

            maybe_sn(nn.Conv2d(128, 256, 3, 2, 1), use_sn),
            nn.LeakyReLU(0.2),
        )
        self.fc = maybe_sn(nn.Linear(256 * 4 * 4, 1), use_sn)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)
