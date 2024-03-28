from os import PathLike
from pathlib import Path
from typing import List, TypeVar

import mph
import numpy as np
from typing_extensions import Unpack

T = TypeVar("T", int, float)


class Param:
    def __init__(self, name: str, min_val: T, max_val: T):
        self.name = name
        self._min = min_val
        self._max = max_val

    def filter(self, val):
        return min(max(val, self._min), self._max)


class Comsol:
    def __init__(self, model: PathLike, *optim_params: Unpack[Param]) -> None:
        self.client: mph.Client = mph.start()
        self.cell: mph.Model = self.client.load(model)

        self.params_filter = {param.name: param for param in optim_params}

    @property
    def params(self):
        return {param: self.cell.parameter(param) for param in self.params_filter}

    @property
    def data(self):
        self.cell.export("res")
        with open("exports/res.txt", "r") as file:
            lines = file.readlines()
        xy_values = [
            list(map(float, ",".join(line.split()).split(",")))
            for line in lines
            if not line.startswith("%")
        ]
        arr = np.array(xy_values)
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

    def update(self, **kwargs: dict[str, float]):
        for key, value in kwargs.items():
            if key in self.params_filter:
                new_value = self.params_filter[key].filter(value)
                self.cell.parameter(key, new_value)
        self.cell.mesh()

    def study(self):
        self.cell.mesh()
        self.cell.solve("Study 1")

    def save(self, name: str):
        self.cell.save(Path("models") / "saved" / f"{name}.mph")


# if __name__ == "__main__":
# comsol = Comsol(
#     "models/cell.mph", *[Param(name, 0, 3) for name in ["r", "rr", "p"]]
# )

# print(comsol.data)
# comsol.update(r=0.0015)
# comsol.study()
