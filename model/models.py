import torch
import torch.nn as nn
from torch import Tensor

class Ours(nn.Module):
    def __init__(self, dim: int = 32, depth: int = 3):
        super(Ours, self).__init__()
        self.depth = depth

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), (1, 1), 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), (1, 1), 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.dense = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * (i + 1), dim, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ) for i in range(depth)
        ])

        self.fuse = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(dim * (depth + 1), dim * 2, (3, 3), (1, 1), 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(dim, 1, (1, 1)),
                nn.Tanh()
            )
        )

    def forward(self, ir: Tensor, vi: Tensor) -> Tensor:
        # src = torch.cat([ir, vi], dim=1)
        ir1 = self.encoder1(ir)
        vi1 = self.encoder2(vi)
        x = torch.cat([ir1, vi1], dim=1)
        for i in range(self.depth):
            t = self.dense[i](x)
            x = torch.cat([x, t], dim=1)
        fus = self.fuse(x)
        return fus
