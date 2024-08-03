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
        elif cfg["dataset"]["sampler"] == "field_single":
            out_nums = 1
        elif cfg["dataset"]["sampler"] == "single_point":
            out_nums = 1
        elif cfg["dataset"]["sampler"] == "single_point_wo_rr":
            out_nums = 1
            input_nums = input_nums - 1
        elif cfg["dataset"]["sampler"] == "10_points":
            out_nums = 10
        else:
            raise ValueError(f"Unknown sampler: {cfg['dataset']['sampler']}")
        hidden_layers = []
        hidden_layers_nums = cfg["train"]["hidden_layers"]
        for i in range(hidden_layers_nums):
            hidden_layers.append(nn.Linear(2**(3 + i), 2**(3 + i + 1)))
            hidden_layers.append(nn.ReLU())
        self.model = nn.Sequential(
            nn.Linear(input_nums, 2**3),
            nn.ReLU(),
            *hidden_layers,
            nn.Dropout(cfg["train"]["dropout"]),
            nn.Linear(2**(3 + hidden_layers_nums), out_nums),
        )

    def forward(self, x):
        return self.model(x)
