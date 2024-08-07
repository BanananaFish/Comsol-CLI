from pathlib import Path

import numpy as np
import pandas as pd


def csv_without_header(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    last_comment_line = 0
    for i, line in enumerate(lines, start=1):
        if line.startswith("%"):
            last_comment_line = i
    return lines[last_comment_line:]

def sample_cood(csv_path: Path, frac: float = 0.1):
    # 读取整个文件
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 找到最后一个以%开头的行
    last_comment_line = 0
    for i, line in enumerate(lines):
        if line.startswith("%"):
            last_comment_line = i
    df = pd.read_csv(
        csv_path,
        skiprows=last_comment_line + 1,
        header=None,
        names=lines[last_comment_line].strip("%\n").split(","),
    )
    if frac:
        df = df.sample(frac=0.1)

    arr = df.values
    return arr


def compress_save(arr: np.ndarray, save_path: Path):
    np.savez_compressed(save_path, arr)


def grid_avg(csv_path: Path): ...
