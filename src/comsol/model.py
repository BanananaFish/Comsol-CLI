from torch import nn


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        input_nums = len(cfg["cell"].values())
        if cfg["dataset"]["sampler"] == "four_points":
            out_nums = 4
        elif cfg["dataset"]["sampler"] == "six_points":
            out_nums = 6
        elif cfg["dataset"]["sampler"] == "field":
            out_nums = 2
        else:
            raise ValueError(f"Unknown sampler: {cfg['dataset']['sampler']}")
        self.model = nn.Sequential(
            nn.Linear(input_nums, 2**3),
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
            nn.ReLU(),
            # 能带值必然是正值，ReLU一下
        )

    def forward(self, x):
        return self.model(x)
