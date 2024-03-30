import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, int_nums: int = 3, out_nums: int = 10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int_nums, 2**3),
            nn.ReLU(),
            nn.Linear(2**3, 2**4),
            nn.ReLU(),
            nn.Linear(2**4, 2**5),
            nn.ReLU(),
            nn.Linear(2**5, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**7),
            nn.ReLU(),
            nn.Linear(2**7, out_nums),
            # 能带值必然是正值，ReLU一下
        )

    def forward(self, x):
        return self.model(x)
