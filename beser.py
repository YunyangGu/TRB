import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from shapely.geometry import LineString, Polygon
from scipy.spatial import distance
from pyswarm import pso
import math

# 修改车道宽度为6
d = 6.0

# 定义车辆大小
vehicle_width = 1.5
vehicle_length = 4.0

# 生成实验车辆的位置
test_vehicle_start_position = [0, random.choice([-d / 4, d / 4])]
test_vehicle_end_position = [150, -test_vehicle_start_position[1]]

# 在两个车道上随机生成4辆车，但排除试验车辆的起始和结束位置
random_positions = [random.uniform(0 + vehicle_length, 150 - vehicle_length * 2) for _ in range(4)]
random_lanes = [random.choice([-d / 4, d / 4]) for _ in range(4)]
rect_centers = list(zip(random_positions, random_lanes))

# 计算一个点到一组矩形的最小距离
def point_to_rects_min_distance(point, rect_centers, rect_length, rect_width):
    min_distance = float('inf')
    for rect_center in rect_centers:
        distance = point_to_rect_min_distance(point, rect_center, rect_length, rect_width)
        min_distance = min(min_distance, distance)
    return min_distance

# 计算一个点到一个矩形的最小距离
def point_to_rect_min_distance(point, rect_center, rect_length, rect_width):
    dx = max(abs(point[0] - rect_center[0]) - rect_length / 2, 0)
    dy = max(abs(point[1] - rect_center[1]) - rect_width / 2, 0)
    return math.sqrt(dx ** 2 + dy ** 2)

# 计算贝塞尔曲线上的点，并在y=0.8处停止
def bezier_curve(P, t):
    n = len(P) - 1
    t = np.reshape(t, (-1, 1))
    C = np.array([math.factorial(n) / (math.factorial(i) * math.factorial(n - i)) for i in range(n+1)])
    T = (1 - t) ** (n - np.arange(n+1)) * t ** np.arange(n+1)
    Path = np.dot(T * C, P)
    Path = Path[Path[:, 1] <= 0.8]  # 只保留y<=0.8的点
    return Path

# 生成贝塞尔曲线的控制点
def generate_control_points():


# 计算曲线长度
def calculate_total_length(P):
    return np.sum(np.sqrt(np.sum(np.diff(P, axis=0)**2, axis=1)))

# 计算曲率
def calculate_curvature(P):
    dx_dt = np.gradient(P[:, 0])
    dy_dt = np.gradient(P[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    return curvature

# 计算偏移
def calculate_offset(P):
    return np.abs(P[:, 1])

# 计算转向角

# 计算拟合误差

# 优化控制点
def optimize_control_points(P0, P3):
    def objective(control_points):
        P1 = [P0[0] + (P3[0] - P0[0]) / 3, control_points[0]]
        P2 = [P0[0] + 2 * (P3[0] - P0[0]) / 3, control_points[1]]
        P = np.array([P0, P1, P2, P3])
        t = np.linspace(0, 1, 101)  # 从0到1均匀生成101个点
        P_fit = bezier_curve(P, t)

        # Calculate the minimum distance to the obstacles for all points on the path
        min_distances = np.array([point_to_rects_min_distance(point, rect_centers, vehicle_length, vehicle_width) for point in P_fit])
        min_distance = np.min(min_distances)

        # Add a penalty to the objective function for small distances to the obstacles
        penalty = 1 / (min_distance - 0.8 + 1e-6) if min_distance < 0.8 else 0

        # Compute the other terms of the objective function as before
        L = calculate_total_length(P_fit)
        curvatures = calculate_curvature(P_fit)
        Q = np.max(curvatures)
        q = np.min(curvatures)
        offsets = calculate_offset(P_fit)
        M = offsets[0]
        N = np.max(offsets)
        turning_angles = calculate_turning_angle(P_fit)
        G = np.max(turning_angles)
        g = np.min(turning_angles)
        I = calculate_fitting_error(P, P_fit)

        # Normalize the features
        L_normalized=lve
        Q_normalized=lve
        q_normalized=lve
        M_normalized=lve
        N_normalized=lve
        G_normalized=lve
        g_normalized=lve
        I_normalized=lve

        return penalty + L_normalized + Q_normalized + q_normalized + M_normalized + N_normalized + G_normalized + g_normalized + I_normalized

    lb = [-d/4, -d/4]
    ub = [d/4, d/4]

    xopt, fopt = pso(objective, lb, ub, maxiter=500)

    return np.array()

# 计算所有障碍物的边界
rects = [Polygon([(rect_center[0] - vehicle_length / 2, rect_center[1] - vehicle_width / 2),
                  (rect_center[0] + vehicle_length / 2, rect_center[1] - vehicle_width / 2),
                  (rect_center[0] + vehicle_length / 2, rect_center[1] + vehicle_width / 2),
                  (rect_center[0] - vehicle_length / 2, rect_center[1] + vehicle_width / 2)])
         for rect_center in rect_centers]

P = generate_control_points()
P = optimize_control_points(P[0], P[-1])
t = np.arange(a,b,c )
Path = bezier_curve(P, t)

fig, ax = plt.subplots(figsize=(20, 10))  # 增大了figure的大小

# 画出所有的车辆
lve