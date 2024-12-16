import numpy as np
import matplotlib.pyplot as plt
from numpy import radians, cos, sin, degrees, linalg

# MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) 
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) 
        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = self.relu(self.final_input)
        return self.final_output  # 输出层为连续值，适合回归

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.relu_derivative(output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_input)

        # 更新权重
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.weights_input_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
        
        # 返回当前损失值 (均方误差)
        return np.mean(output_error**2)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.backward(X, y, output)
            print(output.shape)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)

# 标准化函数
def standardize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1e-8  # 防止标准差为0
    return (X - mean) / std, mean, std

# 还原标准化函数
def inverse_standardize(y, mean, std):
    return y * std + mean

def point_to_line_distance(px, py, x1, y1, x2, y2):
    # 計算點 (px, py) 到線段 (x1, y1) - (x2, y2) 的距離
    line_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])
    line_len = np.dot(line_vec, line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)
    projection = np.dot(point_vec, line_vec) / line_len
    projection = max(0, min(1, projection))
    closest_point = np.array([x1, y1]) + projection * line_vec
    return np.linalg.norm(np.array([px, py]) - closest_point)
def calculate_distances(current_position, phi, boundary_points):
    x, y = current_position
    
    # 計算車輛的前方、右方和左方座標
    front_x, front_y = x + cos(radians(phi)), y + sin(radians(phi))
    right_x, right_y = x + cos(radians(phi + 45)), y + sin(radians(phi + 45))
    left_x, left_y = x + cos(radians(phi - 45)), y + sin(radians(phi - 45))

    # 初始化最小距離
    front_distance = float('inf')
    right_distance = float('inf')
    left_distance = float('inf')

    # 遍歷每條邊界線段，計算到線段的最短距離，並找出最小距離
    for i in range(len(boundary_points) - 1):
        x1, y1 = boundary_points[i]
        x2, y2 = boundary_points[i + 1]
        
        # 計算車輛前方、右方和左方到這條線段的距離，並找出最小距離
        front_distance = min(front_distance, point_to_line_distance(front_x, front_y, x1, y1, x2, y2))
        right_distance = min(right_distance, point_to_line_distance(right_x, right_y, x1, y1, x2, y2))
        left_distance = min(left_distance, point_to_line_distance(left_x, left_y, x1, y1, x2, y2))

    return front_distance, right_distance, left_distance

def is_out_of_bounds(position, phi, boundary_points, end_position_range, boundary_tolerance=1):
    front_distance, right_distance, left_distance = calculate_distances(position, phi, boundary_points)
    x, y = position
    in_end_zone = end_position_range[0] <= x <= end_position_range[1] and end_position_range[1] <= y <= end_position_range[2]

    return front_distance < boundary_tolerance or right_distance < boundary_tolerance or left_distance < boundary_tolerance or in_end_zone


def update_position(x, y, phi, theta, car_length=6, speed=0.5):
    x_new = x + speed * (cos(radians(phi)) + sin(radians(theta)) * sin(radians(phi)))
    y_new = y + speed * (sin(radians(phi)) - sin(radians(theta)) * cos(radians(phi)))
    phi_new = phi - degrees(sin(radians(theta)) * 2 / car_length)
    return x_new, y_new, phi_new

# 模拟主函数
def main():
    np.random.seed(9)

    # 加载边界点
    boundary_points = np.loadtxt("dataset/edge.txt", delimiter=',', skiprows=1)[2:]

    # 加载数据
    data = np.genfromtxt("dataset/train4dAll.txt", delimiter=' ', dtype=float)
    X = data[:, :3]
    y = data[:, 3].reshape(-1, 1)

    # 对特征和目标进行标准化
    X, X_mean, X_std = standardize_features(X)
    y, y_mean, y_std = standardize_features(y)

    # 初始化 MLP 模型
    mlp = MLP(input_size=3, hidden_size=3, output_size=1, learning_rate=0.001)
    mlp.train(X, y, epochs=1000)
    # return
    # 模拟车辆运动
    x, y, phi = 0, 0, 90  # 初始位置和角度
    end_position_range = (18, 30, 37, 42)
    positions = [(x, y)]
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制边界
    ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'b-', label='Boundary')

    for step in range(500):
        # 计算距离特征
        front_distance, right_distance, left_distance = calculate_distances((x, y), phi, boundary_points)
        distances = np.array([[front_distance, right_distance, left_distance]])
        distances = (distances - X_mean) / X_std  # 标准化输入特征

        # 使用 MLP 预测转向角度（标准化输出）
        theta_standardized = mlp.predict(distances)[0][0]
        theta = float(inverse_standardize(theta_standardized, y_mean, y_std))  # 还原并转换为标量
        theta = np.clip(theta, -40, 40)  # 限制角度范围

        # 更新车辆位置
        x, y, phi = update_position(x, y, phi, theta)
        positions.append((x, y))
        print(f"Step {step}: Position: ({x:.2f}, {y:.2f}), Angle: {phi:.2f}, Theta: {theta:.2f}")

        if is_out_of_bounds((x, y), phi, boundary_points, end_position_range, boundary_tolerance=1):
            print("Vehicle stopped due to boundary or end point reached.")
            break

    # 绘制车辆轨迹
    x_vals, y_vals = zip(*positions)
    ax.plot(x_vals, y_vals, 'r-', alpha=0.6, label='Vehicle Path')
    ax.plot([18, 30], [40, 37], 'r-', linewidth=2, label='End Line')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Vehicle Trajectory")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
