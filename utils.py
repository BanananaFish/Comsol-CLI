from itertools import product
import numpy as np
import yaml

from interface import Param


class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __delitem__(self, key):
        del self.config[key]

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return iter(self.config)

    def __len__(self):
        return len(self.config)

    def __str__(self):
        return str(self.config)

    def __repr__(self):
        return repr(self.config)

    @property
    def params(self):
        res = []
        for name, data in self.config["cell"].items():
            res.append(Param(name, data["min"], data["max"]))
        return res

    @property
    def tasks(self):
        ranged_dict = {
            k: np.arange(*map(float, (v["min"], v["max"], v["step"])))
            for k, v in self.config["cell"].items()
        }
        params = list(ranged_dict.keys())
        values = list(ranged_dict.values())

        res = [
            {params[i]: val for i, val in enumerate(vals)} for vals in product(*values)
        ]

        return res
