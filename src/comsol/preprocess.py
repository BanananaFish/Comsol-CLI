import numpy as np
from pathlib import Path
import pickle

from comsol.utils import Config

def grid_avg(fields, y_parts=10):
    # Sort fields by y value
    fields = fields[fields[:,1].argsort()]
    # Split fields into y_parts parts
    split_fields = np.array_split(fields, y_parts)
    # Calculate the average of field_value for each part
    avg_values = [np.mean(part[:,2]) for part in split_fields]
    return avg_values

def central_points(exp):
    points = []
    fields = list(exp.glob("*.npz"))
    fields.sort(key=lambda x: x.stem)
    for field in fields:
        x, k = int(field.stem[5]), int(field.stem.split("-")[-1].replace(".npz", ""))
        fields = np.load(field)["arr_0"]
    return points

def run(src, dst):
    sample_src = Path(src) / "sampled"
    cfg_src = Path(src) / "cfg"
    exp_datas = list(sample_src.glob("study_*"))
    exp_datas.sort(key=lambda x: x.stem)
    param_datas = list(cfg_src.rglob("*.yaml"))
    param_datas.sort(key=lambda x: (x.parent, x.stem))
    
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)
    for i, (exp_data, param_data) in enumerate(zip(exp_datas, param_datas)):
        curr_data = {}
        for x in range(0, 6):
            for k in range(1, 21):
                fields = np.load(exp_data / f"flied{x}-{k}.npz")["arr_0"]
                avg_values = grid_avg(fields)
                # print(x, k, avg_values)
                curr_data[(x, k)] = avg_values
        with open(dst_path / f"exp_{i:05d}.pkl", "wb") as f:
            pickle.dump(curr_data, f)

if __name__ == "__main__":
    run("exports/slit-5-28", "exports/slit-5-28-avg")