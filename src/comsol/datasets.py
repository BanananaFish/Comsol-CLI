from pathlib import Path
import random
from torch.utils.data import Dataset
import numpy as np
import torch

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
        if cfg["train"]["params_norm_dict"] is None:
            raise ValueError("params_regress must be provided")
        else:
            self.params_regress = cfg["train"]["params_norm_dict"]

    def __len__(self):
        return len(self.exp_datas)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor] | None:
        """fielddataset getitem

        Args:
            idx (int): exp index

        Returns:
            tuple: (params, (mse, grater_mean))
        """
        assert int(self.param_datas[idx].parent.stem.split("_")[-1]) == int(self.exp_datas[idx].stem.split("_")[-1]) - 1
        ava_points = self.central_points(self.exp_datas[idx])
        params = Config(self.param_datas[idx])["curr_task"]
        bds = [self.get_bd_data_by_x(x, k) for x, k, _ in ava_points]
        selected_points = self.select_min_error_points(ava_points, bds)
        bd_mean = np.mean([p[3] for p in selected_points])
        mse = np.mean([(p[3] - bd_mean) ** 2 for p in selected_points])
        grater_mean = np.mean([p[2] for p in selected_points])
        if not selected_points:
            print(f"bad data: {idx=}, {self.exp_datas[idx]}")
            new_idx = random.ranint(0, len(self))
            return self[new_idx]
        
        return (
            torch.tensor(list(self.norm_params(params).values()), dtype=torch.float32),
            torch.tensor([self.norm_mse(mse), grater_mean], dtype=torch.float32)
        )
        
    def norm_params(self, params):
        normed = {k: v / float(self.params_regress[k]) for k, v in params.items()}
        return normed
        
    def denorm_params(self, params):
        denormed = {k: v * float(self.params_regress[k]) for k, v in params.items()}
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
            x, k = int(field.stem[5]), int(field.stem.split("-")[-1].replace(".npz", ""))
            fields = np.load(field)["arr_0"]
            grater = self.central(fields)
            if grater:
                # field 0-1, field 0-2 ..., so k needs to -1
                points.append((x, k - 1, grater))
        return points
    
    def select_min_error_points(self, points, bds) -> list[tuple[int, int, float, float]]:
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
        fields = fields[fields[:,1].argsort()]
        
        # Split fields into y_parts parts
        split_fields = np.array_split(fields, y_parts)
        
        # Calculate the average of field_value for each part
        avg_values = [np.mean(part[:,2]) for part in split_fields]
        
        return np.array(avg_values)
        
    def central(self, fields):
        grad_avgs = self.grid_avg(fields)
        
        hf = len(grad_avgs)//2
        central_val = grad_avgs[hf]
        else_val = np.mean(np.concatenate([grad_avgs[:hf], grad_avgs[hf+1:]]))
        grater = (central_val - else_val) / else_val
        if grater > self.threshold:
            return grater
        else:
            return None


