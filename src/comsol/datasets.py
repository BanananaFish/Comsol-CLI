from pathlib import Path
import random
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import torch
from sklearn.preprocessing import MinMaxScaler


from comsol.utils import Config
from comsol.csvparse import csv_without_header


class FieldDataset(Dataset):
    def __init__(self, exps: str, cfg: Config):
        if isinstance(exps, str):
            exps_path = Path(exps)
        field_data_path = exps_path / "sampled"
        param_data_path = exps_path / "cfg"
        self.bd_datas = list(field_data_path.rglob("bd.csv"))
        self.bd_datas.sort(key=lambda x: x.parent)
        self.field_per_bd = len(csv_without_header(self.bd_datas[0]))
        self.exp_datas = list(field_data_path.glob("study_*"))
        self.exp_datas.sort(key=lambda x: x.stem)
        self.param_datas = list(param_data_path.rglob("*.yaml"))
        self.param_datas.sort(key=lambda x: (x.parent, x.stem))

        self.threshold = float(cfg["train"]["threshold"])
        self.mse_norm = float(cfg["train"]["mse_norm"])
        self.params = cfg["cell"]

        self.cfg = cfg
        if cfg["train"]["params_norm_dict"] is None:
            raise ValueError("params_regress must be provided")
        else:
            self.params_regress = cfg["train"]["params_norm_dict"]

    def __len__(self):
        return min(len(self.exp_datas), len(self.param_datas))

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """fielddataset getitem

        Args:
            idx (int): exp index

        Returns:
            tuple: (params, (mse, finall_mean))
        """
        assert (
            int(self.param_datas[idx].parent.stem.split("_")[-1])
            == int(self.exp_datas[idx].stem.split("_")[-1]) - 1
        )
        ava_points = self.central_points(self.exp_datas[idx])
        params = Config(self.param_datas[idx])["curr_task"]
        if self.cfg["dataset"]["sampler"] in ("single_point", "single_point_wo_rr"):
            if ava_points and ava_points[0][0] == 0:
                if self.cfg["dataset"]["bad_data_filter"]:
                    x_idxs = [p[0] for p in ava_points]
                    if 5 in x_idxs:
                        return (
                            torch.tensor(self.norm_params(params), dtype=torch.float32),
                            torch.tensor([ava_points[0][2]], dtype=torch.float32),
                        )
                    else:
                        # print(f"{idx=}, cant find idx-5 points, fallback to random")
                        new_idx = random.randint(0, len(self))
                        return self[new_idx]
                else:
                    return (
                        torch.tensor(self.norm_params(params), dtype=torch.float32),
                        torch.tensor([ava_points[0][2]], dtype=torch.float32),
                    )
            else:
                print(f"{idx=}, cant find idx-0 points, fallback to random")
                new_idx = random.randint(0, len(self))
                return self[new_idx]
        bds = [self.get_bd_data_by_x(x, k) for x, k, _ in ava_points]
        selected_points = self.select_min_error_points(ava_points, bds)
        bd_mean = np.mean([p[3] for p in selected_points])
        mse = np.mean([(p[3] - bd_mean) ** 2 for p in selected_points])
        finall_mean = np.mean([p[2] for p in selected_points])
        if not selected_points:
            print(f"bad data: {idx=}, {self.exp_datas[idx]}")
            new_idx = random.randint(0, len(self))
            return self[new_idx]

        return (
            torch.tensor(self.norm_params(params), dtype=torch.float32),
            torch.tensor([self.norm_mse(mse), finall_mean], dtype=torch.float32),
        )

    def norm_params(self, params):
        # normed = {k: v / float(self.params_regress[k]) for k, v in params.items()}
        normed = []
        for k, v in params.items():
            if self.cfg["dataset"]["sampler"] == "single_point_wo_rr" and k == "rr":
                continue
            curr_max, curr_min = (
                float(self.params[k]["max"]),
                float(self.params[k]["min"]),
            )
            normed.append((v - curr_min) / (curr_max - curr_min))
        return normed

    def denorm_params(self, params):
        denormed = []
        for k, v in params.items():
            curr_max, curr_min = (
                float(self.params[k]["max"]),
                float(self.params[k]["min"]),
            )
            denormed.append(v * (curr_max - curr_min) + curr_min)
        return denormed

    def norm_mse(self, mse):
        return mse / self.mse_norm

    def denorm_mse(self, mse):
        return mse * self.mse_norm

    def get_fields(self, exp, x, k):
        fields = np.load(self.exp_datas[exp] / f"flied{x}-{k}.npz")["arr_0"]
        params = Config(self.param_datas[exp])["curr_task"]
        return (
            fields,
            params,
        )

    def central_points(self, exp):
        points = []
        fields = list(exp.glob("*.npz"))
        fields.sort(key=lambda x: x.stem)
        for field in fields:
            x, k = (
                int(field.stem[5]),
                int(field.stem.split("-")[-1].replace(".npz", "")),
            )
            fields = np.load(field)["arr_0"]
            grater = self.central(fields)
            if grater:
                # field 0-1, field 0-2 ..., so k needs to -1
                points.append((x, k - 1, grater))
        return points

    def select_min_error_points(
        self, points, bds
    ) -> list[tuple[int, int, float, float]]:
        bd_mean = np.mean([bd[1] for bd in bds])
        point_per_x = {}
        for point, bd in zip(points, bds):
            x, k, grater = point
            curr_point = point_per_x.get(x, None)
            if curr_point is None:
                point_per_x[x] = (x, k, grater, bd[1])
            else:
                error = (bd[1] - bd_mean) ** 2
                curr_error = (curr_point[3] - bd_mean) ** 2
                if error < curr_error:
                    point_per_x[x] = (x, k, grater, bd[1])
        return [(x, k, grater, bd) for x, k, grater, bd in point_per_x.values()]

    def get_bd_data_by_x(self, x, k):
        bd_lines = csv_without_header(self.bd_datas[x])
        x_bd = bd_lines[x * 20 + k]
        return x * 0.1, float(x_bd.split(",")[1])

    def grid_avg(self, fields, y_parts=10):
        # Sort fields by y value
        fields = fields[fields[:, 1].argsort()]

        # Split fields into y_parts parts
        split_fields = np.array_split(fields, y_parts)

        # Calculate the average of field_value for each part
        avg_values = [np.mean(part[:, 2]) for part in split_fields]

        return np.array(avg_values)

    def central(self, fields):
        grad_avgs = self.grid_avg(fields)

        hf = len(grad_avgs) // 2
        central_val = grad_avgs[hf]
        if self.cfg["dataset"]["feature"] == "mse":
            error = np.array(grad_avgs) - central_val
            mse = (error / central_val).var()
            if mse < self.threshold:
                return mse
            else:
                return None
        elif self.cfg["dataset"]["feature"] == "percentage":
            else_val = np.mean(np.concatenate([grad_avgs[:hf], grad_avgs[hf + 1 :]]))
            res = (central_val - else_val) / else_val
            if res > self.threshold:
                return res
            else:
                return None
        else:
            raise ValueError("Config.dataset.feature is not set!")


class BDDataset(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        self.params_name = [name for name in self.data.columns if "path" not in name and "bd" not in name]
        self.scaler = MinMaxScaler()
        self.data[self.params_name] = self.scaler.fit_transform(
            self.data[self.params_name]
        )
        self.freq_max = 7800

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """fielddataset getitem

        Args:
            idx (int): exp index

        Returns:
            tuple: (params, (mse, finall_mean))
        """
        row = self.data.iloc[idx]
        bd_columns = [f'bd_{i}' for i in range(10)]
        bd = row[bd_columns].values.astype(np.float32)
        params = row[self.params_name].values.astype(np.float32)
        return (
            torch.tensor(params, dtype=torch.float32),
            torch.tensor(bd, dtype=torch.float32) / self.freq_max,
        )

    def get_bds(self, bd_path):
        bd_csv = csv_without_header(bd_path)
        d = defaultdict(list)
        for row in bd_csv:
            x, y = row.replace("\n", "").split(",")
            x, y = float(x), float(y)
            d[x].append(y)
        for y in d.values():
            y.sort()
        max_x, min_x = max(d.keys()), min(d.keys())
        ret = []
        left = right = 0
        for i in range(10):
            if i % 2 == 0:
                ret.append(d[min_x][left])
                left += 1
            else:
                ret.append(d[max_x][right])
                right += 1
        return ret

    def denorm_params(self, params):
        if isinstance(params, torch.Tensor):
            params = params.numpy()
        params = params.reshape(1, -1)
        return self.scaler.inverse_transform(params)

    def denorm_bd(self, bd):
        return bd * self.freq_max

    def denorm(self, normed_data):
        params, bd = normed_data
        return self.denorm_params(params), self.denorm_bd(bd)
