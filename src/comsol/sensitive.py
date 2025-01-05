import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


# 读取CSV文件
data = pd.read_csv('exports/slit-9-1/params_del_bad.csv')
params_name = ['R', 'r', 'slit', 't', 'w']
params = data[params_name].values
responses = data[[f'bd_{i}' for i in range(10)]].values
# 对参数进行归一化
scaler_params = MinMaxScaler()
params_normalized = scaler_params.fit_transform(params)

# 对响应数据进行归一化（如果需要）
scaler_responses = MinMaxScaler()
responses_normalized = scaler_responses.fit_transform(responses)

# 计算灵敏度
def calculate_sensitivity(params, responses):
    num_params = params.shape[1]
    sensitivities = []
    
    for i in range(num_params):
        # 使用线性回归计算灵敏度
        slope, _, _, _, _ = stats.linregress(params[:, i], responses.mean(axis=1))
        sensitivities.append(slope)
    
    return sensitivities

sensitivities = calculate_sensitivity(params_normalized, responses_normalized)

print('Sensitivity:', {name: sensitivity for name, sensitivity in zip(params_name, sensitivities)})
