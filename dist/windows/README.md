# 使用说明

## 下载
1. 右侧点击`Release`
2. 下载可执行文件（不要下载源码）

## 执行
请在终端中执行（CMD或PowerShell）
```powershell
comsol_CLI.exe --help

# 获取实验结果
comsol_CLI.exe run --model <MPH path> --config <config yaml> [--dump]

# 训练
comsol_CLI.exe train --saved <pickles path> --config <config yaml> --ckpt_path <ckpt dir>

# 搜索最优参数
comsol_CLI.exe ga --ckpt <model.pth path> --saved <pickles path>
```

## 自己打包
```powershell
pyinstaller -F --specpath dist/windows src/comsol/cmdline.py
```