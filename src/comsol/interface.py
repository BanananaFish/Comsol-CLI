from __future__ import annotations

import pickle
import shutil
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import List, TypeVar

import mph
import numpy as np
import yaml
from typing_extensions import deprecated

from comsol.console import console
from comsol.csvparse import compress_save, grid_avg, sample_cood

T = TypeVar("T", int, float)


class Param:
    def __init__(self, name: str, min_val: T, max_val: T):
        self.name = name
        self._min = min_val
        self._max = max_val

    def __repr__(self) -> str:
        return f"<Param #{self.name} ({self._min}, {self._max})>"

    def filter(self, val):
        return min(max(val, self._min), self._max)


class Comsol:
    def __init__(
        self, model: PathLike | str, export_dir: PathLike | str, *optim_params: Param
    ) -> None:
        self.client: mph.Client = mph.start()
        self.cell: mph.Model = self.client.load(model)

        self.export_dir = Path(export_dir)

        self.params_filter = {param.name: param for param in optim_params}

        self.study_count = 0

    @deprecated("parse_res is deprecated")
    def parse_res(self):
        self.cell.export()
        # with open(self.export_file, "r") as file:
        #     lines = file.readlines()
        # xy_values = [
        #     list(map(float, ",".join(line.split()).split(",")))
        #     for line in lines
        #     if not line.startswith("%")
        # ]
        # arr = np.array(xy_values)
        # return arr

    @property
    def params(self):
        return {param: self.cell.parameter(param) for param in self.params_filter}

    @property
    @deprecated("data property is deprecated")
    def data(self):
        self.cell.export()
        # if Path(self.export_file).exists():
        #     arr = self.parse_res()
        #     arr_sorted = arr[arr[:, 0].argsort()]  # 按照x值对arr进行排序

        #     min_values: dict[
        #         float, List[float]
        #     ] = {}  # 初始化一个空的字典来存储每个x值的最小的两个数

        #     for x, y in arr_sorted:
        #         if x not in min_values:
        #             min_values[x] = [y]
        #         else:
        #             if len(min_values[x]) < 2:
        #                 min_values[x].append(y)
        #             else:
        #                 max_value = max(min_values[x])
        #                 if y < max_value:
        #                     min_values[x].remove(max_value)
        #                     min_values[x].append(y)
        #     return min_values

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params_filter:
                new_value = self.params_filter[key].filter(value)
                self.cell.parameter(key, new_value)
        console.log(f"curr_params: {self.params}")
        self.cell.mesh()

    def study(self):
        console.log(f"# {self.study_count + 1} Solving...")
        self.cell.solve()
        self.study_count += 1

    def save_cfg(self, cfg, curr_task):
        cfg_ = deepcopy(cfg)
        dest_dir = self.export_dir / "cfg" / f"study_{self.study_count:05d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        cfg_.config["curr_task"] = {k: float(v) for k, v in curr_task.items()}
        cfg_.dump(dest_dir / f"cfg_{self.study_count:05d}.yaml")
        console.log(f"Config saved to {dest_dir}")

    @deprecated("save_pkl is deprecated, use save_raw_data / save_avg_data instead")
    def save_pkl(self):
        self.cell.export()
        dest_dir = self.export_dir / "raw" / f"study_{self.study_count:05d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / f"res_{self.study_count:05d}.pkl", "wb") as f:
            pickle.dump((self.params, self.parse_res()), f)
        console.log(f"Results saved to {dest_dir}")

    def save_raw_data(self):
        dest_dir = self.export_dir / "raw" / f"study_{self.study_count:05d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        export_tasks = self.cell.exports()
        for name in export_tasks:
            csv = ".." / dest_dir / f"{name}.csv"
            console.log(f"Results({name}) saved to {csv}")
            self.cell.export(name, csv)

    def save_avg_data(self, avg_list: List[str] = ["flied"]):
        raise NotImplementedError("save_avg_data is not implemented")
        dest_dir = self.export_dir / "raw" / f"study_{self.study_count:05d}"
        tmp_dir = self.export_dir / "avg" / "tmp"
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        export_tasks = self.cell.exports()
        for task in export_tasks:
            csv_name = f"{task}.csv"
            self.cell.export(task, (tmp_dir / csv_name).absolute())
            if task in avg_list:
                grid_avg(tmp_dir / csv_name)
                console.log(f"Results({task}) cal grid avg")
            shutil.copy(tmp_dir / csv_name, dest_dir / csv_name)
            console.log(f"Results({task}) saved to {dest_dir}")

    def save_sampled_data(
        self, frac: float, sample_keys: List[str], console=console, progress=None
    ):
        dest_dir = self.export_dir / "sampled" / f"study_{self.study_count:05d}"
        tmp_dir = self.export_dir / "tmp"
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        export_tasks = self.cell.exports()
        if progress:
            sampled_task = progress.add_task(
                "[light_cyan3]Sample", total=len(export_tasks)
            )
        for task in export_tasks:
            csv_name = f"{task}.csv"
            # 导出的工作路径使用的是 cell 模型的路径，所以目标需要使用绝对路径
            self.cell.export(task, (tmp_dir / csv_name).absolute())
            if any(sample_key in task for sample_key in sample_keys):
                arr = sample_cood(tmp_dir / csv_name)
                compress_save(arr, dest_dir / f"{task}.npz")
                # console.log(f"Results({task}) sampled! frac: {frac:.3f}")
            else:
                shutil.copy(tmp_dir / csv_name, dest_dir / csv_name)
                # console.log(f"Results({task}) skip sample, saved to {dest_dir}")

            if progress:
                progress.update(
                    sampled_task,
                    advance=1,
                    description=f"[light_cyan3]Sample: {task}",
                )
        console.log(f"Sampled saved to {dest_dir}")

        if progress:
            progress.stop_task(sampled_task)
            progress.remove_task(sampled_task)

    def dump(self):
        dest = Path("models") / "saved" / f"cell_{self.study_count:05d}.mph"
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.cell.save(dest)
        console.log(f"Model dumped to {dest}")
