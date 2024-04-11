import pickle
from os import PathLike
from pathlib import Path
from typing import List, TypeVar

import mph
import numpy as np

from comsol.console import console

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
    def __init__(self, model: PathLike | str, *optim_params: Param) -> None:
        self.client: mph.Client = mph.start()
        self.cell: mph.Model = self.client.load(model)

        self.export_file = Path("exports") / "res.txt"

        self.params_filter = {param.name: param for param in optim_params}

        self.study_count = 0

    def parse_res(self):
        self.cell.export()
        with open(self.export_file, "r") as file:
            lines = file.readlines()
        xy_values = [
            list(map(float, ",".join(line.split()).split(",")))
            for line in lines
            if not line.startswith("%")
        ]
        arr = np.array(xy_values)
        return arr

    @property
    def params(self):
        return {param: self.cell.parameter(param) for param in self.params_filter}

    @property
    def data(self):
        self.cell.export()
        if Path(self.export_file).exists():
            arr = self.parse_res()
            arr_sorted = arr[arr[:, 0].argsort()]  # 按照x值对arr进行排序

            min_values: dict[
                float, List[float]
            ] = {}  # 初始化一个空的字典来存储每个x值的最小的两个数

            for x, y in arr_sorted:
                if x not in min_values:
                    min_values[x] = [y]
                else:
                    if len(min_values[x]) < 2:
                        min_values[x].append(y)
                    else:
                        max_value = max(min_values[x])
                        if y < max_value:
                            min_values[x].remove(max_value)
                            min_values[x].append(y)
            return min_values

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params_filter:
                new_value = self.params_filter[key].filter(value)
                self.cell.parameter(key, new_value)
        self.cell.mesh()
        console.log(self.params)

    def study(self):
        console.log(f"# {self.study_count + 1} Solving...")
        self.cell.solve()
        self.cell.export()
        self.study_count += 1

    def save(self):
        self.cell.export()
        dest = Path("exports") / "saved" / f"res_{self.study_count:05d}.pkl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.export_file.exists():
            with open(dest, "wb") as f:
                pickle.dump((self.params, self.parse_res()), f)
            console.log(f"Results saved to {dest}")

    def save_raw_data(self):
        dest_dir = Path("exports") / "raw" / f"study_{self.study_count:05d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        export_tasks = self.cell.exports()
        for name in export_tasks:
            csv = ".." / dest_dir / f"{name}.csv"
            console.log(f"Results({name}) saved to {csv}")
            self.cell.export(name, csv)

    def dump(self):
        dest = Path("models") / "saved" / f"cell_{self.study_count:05d}.mph"
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.cell.save(dest)
        console.log(f"Model dumped to {dest}")


if __name__ == "__main__":
    comsol = Comsol(
        "models/cell.mph", *[Param(name, 0, 3) for name in ["r", "rr", "p"]]
    )

    console.log(comsol.data)
    comsol.update(r=0.0015)
    comsol.study()
    comsol.dump()
