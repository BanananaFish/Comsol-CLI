import pickle
from datetime import datetime
from itertools import product
from os import PathLike
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.progress import Progress
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from comsol.console import console
from comsol.interface import Param


class Config:
    def __init__(self, config_file):
        with open(config_file, "r", encoding="utf-8") as f:
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
    class EarlyStop(Exception):
        pass

    def __init__(self, dataset, model, cfg: Config, ckpt_path):
        self.model = model
        self.cfg = cfg

        self.epoch: int = cfg["train"]["epoch"]
        self.batch_size: int = cfg["train"]["batch_size"]
        self.lr: int = cfg["train"]["lr"]
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.best_loss = float("inf")
        self.best_ckpt = model.state_dict()
        self.stuck_count = 0
        self.early_stop = cfg["train"]["early_stop"]

        self.ckpt_path = ckpt_path

        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size
        train_data, test_data = random_split(dataset, (train_size, test_size))
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=6)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, num_workers=6)

        self.start_time = datetime.now()

    @staticmethod
    def to_cuda(obj):
        if torch.cuda.is_available():
            return obj.cuda()
        return obj

    def train(self):
        self.model.train()
        self.model = self.to_cuda(self.model)
        with Progress(console=console) as progress:
            train_epoch_task = progress.add_task("[yellow]Training", total=self.epoch)
            train_it_task = progress.add_task(
                "[yellow3]Iteration", total=len(self.train_loader)
            )
            for epoch in range(1, self.epoch + 1):
                progress.update(train_epoch_task, advance=1)
                for i, (x, y) in enumerate(self.train_loader):
                    progress.update(train_it_task, advance=1)
                    x, y = self.to_cuda(x), self.to_cuda(y)
                    self.optimizer.zero_grad()
                    y_pred = self.model(x)
                    y_pred = torch.mul(y_pred, torch.tensor(self.cfg["train"]["loss_weight"]))
                    loss = self.loss(y_pred, y)
                    loss.backward()
                    self.optimizer.step()
                    if i % 50 == 0:
                        console.log(
                            f"Epoch [{epoch}/{self.cfg['train']['epoch']}], iter {i}, loss: {loss.item():.6f}"
                        )
                progress.stop_task(train_it_task)
                progress.remove_task(train_it_task)
                self.test()
            self.save_ckpt("lastest")
            console.log(f"Training finished, best loss: {self.best_loss:.6f}")
            self.save_ckpt(f"best_loss_{self.best_loss:.6f}", best=True)

    def test(self):
        self.model.eval()
        losses = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = self.to_cuda(x), self.to_cuda(y)
                y_pred = self.model(x)
                losses += self.loss(y_pred, y)
        now_loss = losses / len(self.test_loader)
        console.log(f"Test loss: {now_loss:.6f}")
        if now_loss < self.best_loss:
            self.stuck_count = 0
            self.best_loss = now_loss
            self.best_ckpt = self.model.state_dict()
        else:
            self.stuck_count += 1
            if self.stuck_count >= self.early_stop:
                raise self.EarlyStop

    def save_ckpt(self, name, best=False):
        ckpt_path = Path(self.ckpt_path) / f"{self.start_time:%Y.%m.%d_%H.%M.%S}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        pth_path = ckpt_path / f"{name}.pth"
        cfg_path = ckpt_path / f"config.yaml"
        if not cfg_path.exists():
            self.cfg.dump(cfg_path)
            console.log(f"Dumped config to {cfg_path}")
        if best:
            torch.save(self.best_ckpt, pth_path)
        else:
            torch.save(self.model.state_dict(), pth_path)
        console.log(f"Saved model to {pth_path}")


class BandDataset(Dataset):
    Bs_filter = set(("B0_1", "B0_11"))

    @staticmethod
    def get_Bs(res_arr, sampler):
        if sampler == "four_points":
            res = BandDataset.four_points_sampler(res_arr)
        elif sampler == "six_points":
            res = BandDataset.six_points_sampler(res_arr)
        elif sampler == "mid":
            res = BandDataset.mid_sampler(res_arr)
        else:
            raise ValueError(f"Sampler {sampler} not found")
        return res

    @staticmethod
    def four_points_sampler(res_arr):
        x_0 = 0
        x_1 = res_arr[:, 0][len(res_arr) // 3]
        res_arr_0 = res_arr[res_arr[:, 0] == x_0]
        res_arr_1 = res_arr[res_arr[:, 0] == x_1]

        # 按照y的值进行排序
        res_arr_0 = res_arr_0[res_arr_0[:, 1].argsort()]
        res_arr_1 = res_arr_1[res_arr_1[:, 1].argsort()]

        # line0_0, line1_0, line0_1, line1_1
        # 1 . 3
        # . . .
        # 0 . 2
        res = np.array(
            [res_arr_0[0, 1], res_arr_0[1, 1], res_arr_1[0, 1], res_arr_1[1, 1]]
        )

        assert len(res) == 4, f"res length is {len(res)}, not 4"
        return res

    @staticmethod
    def six_points_sampler(res_arr):
        x_0 = 0
        x_1 = res_arr[:, 0][len(res_arr) // 3]
        x_2 = res_arr[:, 0][(len(res_arr) * 2) // 3]
        res_arr_0 = res_arr[res_arr[:, 0] == x_0]
        res_arr_1 = res_arr[res_arr[:, 0] == x_1]
        res_arr_2 = res_arr[res_arr[:, 0] == x_2]

        # 按照y的值进行排序
        res_arr_0 = res_arr_0[res_arr_0[:, 1].argsort()]
        res_arr_1 = res_arr_1[res_arr_1[:, 1].argsort()]
        res_arr_2 = res_arr_2[res_arr_2[:, 1].argsort()]

        # line0_0, line1_0, line0_1, line1_1
        # 1 . 3 . 5
        # . . . . .
        # 0 . 2 . 4
        res = np.array(
            [
                res_arr_0[0, 1],
                res_arr_0[1, 1],
                res_arr_1[0, 1],
                res_arr_1[1, 1],
                res_arr_2[0, 1],
                res_arr_2[1, 1],
            ]
        )

        assert len(res) == 6, f"res length is {len(res)}, not 6"
        return res

    @staticmethod
    def mid_sampler(res_arr):
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

    def __init__(self, saved_path: PathLike | str, cfg: Config):
        self.params_mins = []
        self.params_maxs = []

        params_list = []
        res_arr_list = []
        # read raw data
        for pkl in Path(saved_path).glob("*.pkl"):
            with open(pkl, "rb") as f:
                data: tuple[dict[str, str], np.ndarray] = pickle.load(f)
                params, res = data
                # 把第一个位置的参数归一化
                params_list.append(self.to_rand(params))
                res_arr_list.append(self.get_Bs(res, cfg["dataset"]["sampler"]))

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
