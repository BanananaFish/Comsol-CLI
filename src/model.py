import torch
from torch import nn


class MLP(nn.modules):
    def __init__(self, param_nums: int = 3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(param_nums, 2**4),
            nn.ReLU(),
            nn.Linear(2**4, 2**5),
            nn.ReLU(),
            nn.Linear(2**5, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**7),
            nn.ReLU(),
            nn.Linear(2**7, 2**8),
            nn.ReLU(),
            nn.Linear(2**8, 2**9),
            nn.ReLU(),
            nn.Linear(2**9, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)
