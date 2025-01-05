# 使用说明

## 下载
1. 右侧点击`Release`
2. 下载可执行文件（不要下载源码）

## 目录结构
```bash
.
└── ComsolCLI
    ├── configs (这里放配置文件)
    │   ├── cell.yaml
    │   └── ...
    ├── models (这里放模型文件)
    │   ├── cell.mph
    │   └── ...
    ├── logs (这里放训练结果)
    │   ├── field_ckpt
    │   ├── six_points_ckpt
    │   └── ...
    ├── exps (这里放实验出来的数据，每个实验的目录结构一定要保持有cfg和sample)
    │   ├── exp1
    │   │   ├── cfg
    │   │   │   ├── study_00001
    │   │   │   │   └── *.yaml
    │   │   │   ├── study_00002
    │   │   │   │   └── *.yaml
    │   │   │   └── ...
    │   │   └── sampled
    │   │       ├── study_00001
    │   │       │   └── field*.npz
    │   │       ├── study_00002
    │   │       │   └── field*.npz
    │   │       └── ...
    │   └── ...
    └── comsol_CLI.exe
```

## 执行
请在终端中执行（CMD或PowerShell）
```powershell
comsol_CLI.exe --help

# 获取实验结果
comsol_CLI.exe run --model <MPH path> --config <config yaml> [--dump / --raw / --sample 0.1]
# 例如
comsol_CLI.exe run --model models/cell.mph --config config/cell.yaml --sample 0.1

# 训练
comsol_CLI.exe train --exps <exps path> --config <config yaml> --ckpt_path <ckpt dir>
# 例如
comsol_CLI.exe train --exps exps/exp1 --config config/cell.yaml --ckpt_path ckpt/field_ckpt 

# 搜索最优参数
comsol_CLI.exe ga --ckpt <model.pth path> --saved <pickles path>
# 例如
comsol_CLI.exe train --exps exps/exp1 --config config/cell.yaml --ckpt_path ckpt/field_ckpt 
```

## 自己打包
```powershell
pyinstaller -F --specpath dist/windows src/comsol/cmdline.py
```