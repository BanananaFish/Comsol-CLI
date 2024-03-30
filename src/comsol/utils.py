import pickle
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from numpy.typing import ArrayLike
from rich.progress import track
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from comsol.interface import Param
from comsol.model import MLP


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

    def dump(self, path: PathLike | str):
        with open(path, "w") as f:
            yaml.dump(self.config, f)


class Trainer:
    def __init__(self, dataset, model, cfg: Config):
        self.model = model
        self.cfg = cfg

        self.epoch: int = cfg["train"]["epoch"]
        self.batch_size: int = cfg["train"]["batch_size"]
        self.lr: int = cfg["train"]["lr"]
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.best_loss = float("inf")
        self.best_ckpt = model.state_dict()

        dataset_size = len(dataset)
        train_data, test_data = random_split(
            dataset, (int(dataset_size * 0.8), int(dataset_size * 0.2))
        )
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        self.start_time = datetime.now()

    @staticmethod
    def to_cuda(obj):
        if torch.cuda.is_available():
            return obj.cuda()
        return obj

    def train(self):
        self.model.train()
        self.model = self.to_cuda(self.model)
        for epoch in range(1, self.epoch + 1):
            for i, (x, y) in track(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                description=f"Epoch {epoch}",
                auto_refresh=False,
            ):
                x, y = self.to_cuda(x), self.to_cuda(y)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.3f}")
            self.test()
        self.save_ckpt("lastest")
        logger.info(f"Training finished, best loss: {self.best_loss:.3f}")
        self.save_ckpt(f"best_loss_{self.best_loss:.3f}")

    def test(self):
        self.model.eval()
        losses = 0
        with torch.no_grad():
            for x, y in track(
                self.test_loader, description="Testing", auto_refresh=False
            ):
                x, y = self.to_cuda(x), self.to_cuda(y)
                y_pred = self.model(x)
                losses += self.loss(y_pred, y)
        now_loss = losses / len(self.test_loader)
        logger.info(f"Test loss: {now_loss:.3f}")
        if now_loss < self.best_loss:
            self.best_loss = now_loss
            self.best_ckpt = self.model.state_dict()

    def save_ckpt(self, name):
        ckpt_path = Path(f"ckpt") / f"{self.start_time:%Y.%m.%d_%H.%M.%S}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        pth_path = ckpt_path / f"{name}.pth"
        cfg_path = ckpt_path / f"config.yaml"
        if not cfg_path.exists():
            self.cfg.dump(cfg_path)
            logger.info(f"Dumped config to {cfg_path}")
        torch.save(self.model.state_dict(), pth_path)
        logger.info(f"Saved model to {pth_path}")


class BandDataset(Dataset):
    Bs_filter = set(("B0_1", "B0_11"))

    @staticmethod
    def get_Bs(res_arr):
        Bs: list[tuple[str, float]] = []
        x_mid = res_arr[:, 0][len(res_arr) // 2]
        res_arr_0 = res_arr[res_arr[:, 0] == 0]
        res_arr_mid = res_arr[res_arr[:, 0] == x_mid]

        # 按照y的值进行排序
        res_arr_0 = res_arr_0[res_arr_0[:, 1].argsort()]
        res_arr_mid = res_arr_mid[res_arr_mid[:, 1].argsort()]
        for i in range(1, 13):
            if i % 2 == 0:
                if len(res_arr_mid) > 0:
                    Bs.append((f"Bmid_{i}", res_arr_mid[0, 1]))
                    res_arr_mid = res_arr_mid[1:]
            else:
                if len(res_arr_0) > 0:
                    Bs.append((f"B0_{i}", res_arr_0[0, 1]))
                    res_arr_0 = res_arr_0[1:]

        res = np.array([b[1] for b in Bs if b[0] not in BandDataset.Bs_filter])
        assert len(res) == 10, f"res length is {len(res)}"
        return res

    @staticmethod
    def to_rand(params: dict[str, str]):
        zeta, rr, rrr = map(float, params.values())
        return np.array([zeta / 360, rr, rrr])

    @staticmethod
    def normalization(arr: np.ndarray):
        min = arr.min()
        max = arr.max()
        return (arr - min) / (max - min), min, max

    def denormalization(self, params: np.ndarray, res: np.ndarray):
        for i in range(len(params)):
            params[i] = (
                params[i] * (self.params_maxs[i] - self.params_mins[i])
                + self.params_mins[i]
            )
        res = res * (self.res_max - self.res_min) + self.res_min
        return params, res

    def __init__(self, saved_path: PathLike | str):
        self.params_mins = []
        self.params_maxs = []

        params_list = []
        res_arr_list = []
        # read raw data
        for pkl in Path(saved_path).glob("*.pkl"):
            with open(pkl, "rb") as f:
                data: tuple[dict[str, str], np.ndarray] = pickle.load(f)
                params, res = data
                params_list.append(self.to_rand(params))
                res_arr_list.append(self.get_Bs(res))

        # normalization
        params_arr = np.array(params_list)
        res_arr = np.array(res_arr_list)
        for i in range(params_arr.shape[1]):
            params_arr[:, i], p_min, p_max = self.normalization(params_arr[:, i])
            self.params_mins.append(p_min)
            self.params_maxs.append(p_max)
        res_arr, res_min, res_max = self.normalization(res_arr)
        self.res_min = res_min
        self.res_max = res_max
        self.datas = (params_arr, res_arr)

    def __len__(self):
        return len(self.datas[0])

    def __getitem__(self, idx):
        # return self.datas[idx]
        return (
            torch.tensor(self.datas[0][idx]).float(),
            torch.tensor(self.datas[1][idx]).float(),
        )


if __name__ == "__main__":
    cfg = Config("config/cell2.yaml")
    dataset = BandDataset("exports/saved")
    # print(len(dataset))
    # param, res = dataset[0]
    # print(param, res)
    # print(dataset.denormalization(param.numpy(), res.numpy()))
    # bigs = dataset[0][1]
    # smalls = dataset[0][0]
    # for b in bigs:
    #     print(torch.div(b, 1e10))
    # for s in smalls:
    #     print(torch.mul(s, 1e3))
    model = MLP()
    trainer = Trainer(dataset, model, cfg)
    trainer.train()
