from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from comsol.csvparse import csv_without_header
from comsol.utils import Config


def merge_params_to_csv(cfg_path):
    cfg_path = Path(cfg_path)
    cfgs = list(Path(cfg_path).rglob("*.yaml"))
    cfgs.sort(key=lambda x: x.parent)
    heads = Config(cfgs[0])["curr_task"].keys()
    data = {head: [] for head in heads}
    data["path"] = []
    for c_path in tqdm(cfgs):
        cfg = Config(c_path)
        params = cfg["curr_task"]
        data["path"].append(str(c_path))
        for head in heads:
            data[head].append(params[head])
    df = pd.DataFrame(data)
    df.to_csv(cfg_path.parent / "params.csv", index=False)

def add_bd_path(csv_path):
    df = pd.read_csv(csv_path)
    bd_path = Path(csv_path).parent / "raw"
    bds = list(bd_path.rglob("bd.csv"))
    bds.sort(key=lambda x: x.parent)
    df["bd_path"] = [str(bd) for bd in bds]
    df.to_csv(csv_path, index=False)

def add_bd_data(csv_path, out_path):
    def get_bds(bd_path):
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
    df = pd.read_csv(csv_path)
    # 初始化新的列
    for i in range(10):
        df[f'bd_{i}'] = None

    # 对每一行计算bd并添加到新的列中
    for index, row in tqdm(df.iterrows()):
        bd = get_bds(row["bd_path"])
        for i in range(10):
            df.at[index, f'bd_{i}'] = bd[i]

    # 保存结果
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # merge_params_to_csv("exports/slit-9-1/cfg")
    # add_bd_path("exports/slit-9-1/params.csv")
    add_bd_data("exports/slit-9-1/params.csv", "exports/slit-9-1/params_updated.csv")
    pass