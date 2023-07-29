import numpy as np
import scipy
from scipy.optimize import least_squares
import pandas as pd
from multiprocessing import Pool

# 读取数据文件
data = pd.read_excel()

# 定义贝塞尔曲线函数
def bezier(t, points):
    # 贝塞尔曲线的阶数由控制点的数量决定
    n = len(points) - 1
    # 计算贝塞尔曲线
    return sum(scipy.special.comb(n, i) * (1 - t) ** (n - i) * t ** i * points[i] for i in range(n + 1))

# 定义损失函数
def loss_func(params, t, y):
    # 计算贝塞尔曲线
    curve = bezier(t, params)
    # 计算损失
    loss = np.sum((curve - y) ** 2)
    return loss


def fit_bezier(points, order):
    """lve"""
# 定义一个函数来对每个车辆进行拟合
def fit_for_vehicle(vehicle_data):
     """lve"""

# 定义要尝试的控制点数量范围，注意这里改变了定义，order现在表示控制点的数量，而不是贝塞尔曲线的阶数
order_range = range(4, 7)

if __name__ == '__main__':
    # 创建一个进程池
    with Pool() as pool:
        # 对每个车辆应用该函数
        all_results = pool.map(fit_for_vehicle, [group for _, group in data.groupby('Vehicle ID')])

    # 将结果转换为DataFrame
    results_df = pd.DataFrame([item for sublist in all_results for item in sublist])

    # 输出结果
    print(results_df)

    # 将结果保存到Excel文件
    results_df.to_excel('output.xlsx', index=False, engine='openpyxl')