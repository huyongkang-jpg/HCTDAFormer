import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class PEMS03Visualizer:
    def __init__(self, data_path=None, sample_rate=1):
        """
        初始化PEMS03数据集可视化工具

        参数:
            data_path: 数据文件路径，如果为None则生成模拟数据
            sample_rate: 采样率，用于减少数据量，提高可视化速度
        """
        self.sample_rate = sample_rate
        if data_path:
            self.load_data(data_path)
        else:
            self.generate_sample_data()

        # 计算时间索引
        self._calculate_time_index()

    def load_data(self, data_path):
        """加载PEMS03数据集"""
        # 假设数据是npy格式，形状为(26208, 358, 1)
        try:
            self.data = np.load(data_path)[:, :, 0]  # 去掉最后一维(1)
            print(f"数据加载成功，形状: {self.data.shape}")
        except:
            print("数据加载失败，使用模拟数据...")
            self.generate_sample_data()

    def generate_sample_data(self):
        """生成模拟的PEMS03数据集"""
        # 生成26208个时间点，358个传感器的流量数据
        np.random.seed(42)
        time_steps = 26208
        sensors = 358

        # 基础流量
        base_flow = np.random.normal(1000, 300, size=(time_steps, sensors))

        # 添加日周期模式（早高峰、晚高峰）
        hour_of_day = np.arange(time_steps) % 288  # 每天288个时间步(5分钟间隔)
        daily_pattern = np.sin(2 * np.pi * hour_of_day / 288) * 300
        daily_pattern += np.sin(4 * np.pi * hour_of_day / 288) * 200  # 双周期
        daily_pattern = daily_pattern[:, np.newaxis]  # 扩展维度以广播到所有传感器

        # 添加周周期模式（工作日流量高，周末流量低）
        day_of_week = (np.arange(time_steps) // 288) % 7
        weekly_pattern = np.ones(time_steps)
        weekly_pattern[day_of_week >= 5] = 0.7  # 周末流量降低30%
        weekly_pattern = weekly_pattern[:, np.newaxis]

        # 添加传感器间的相关性（相邻传感器流量相似）
        spatial_correlation = np.random.normal(0, 1, size=sensors)
        spatial_correlation = np.cumsum(spatial_correlation)
        spatial_correlation = spatial_correlation / np.max(np.abs(spatial_correlation)) * 200

        # 组合所有成分
        self.data = base_flow + daily_pattern * weekly_pattern + spatial_correlation

        # 确保流量非负
        self.data = np.maximum(0, self.data)

        print(f"模拟数据生成成功，形状: {self.data.shape}")

    def _calculate_time_index(self):
        """计算时间索引"""
        # 假设数据从2023年1月1日开始，每5分钟一个时间点
        start_date = datetime(2023, 1, 1)
        time_delta = [timedelta(minutes=5 * i) for i in range(self.data.shape[0])]
        self.time_index = np.array([start_date + td for td in time_delta])

        # 采样
        self.time_index = self.time_index[::self.sample_rate]
        self.data = self.data[::self.sample_rate]

    def plot_time_series(self, sensor_indices=[0, 1, 2], days=3):
        """
        绘制指定传感器的时间序列流量图

        参数:
            sensor_indices: 要绘制的传感器索引列表
            days: 要显示的天数
        """
        time_steps = min(288 * days, len(self.time_index))  # 每天288个时间步

        plt.figure(figsize=(15, 8))
        for i, sensor_idx in enumerate(sensor_indices):
            plt.plot(self.time_index[:time_steps],
                     self.data[:time_steps, sensor_idx],
                     label=f'传感器 {sensor_idx}',
                     linewidth=1.5,
                     alpha=0.7)

        plt.title(f'交通流量时间序列（前{days}天）')
        plt.xlabel('日期时间')
        plt.ylabel('流量（辆/5分钟）')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 设置x轴日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.show()

    def plot_daily_pattern(self, sensor_idx=0, days=7):
        """
        绘制日流量模式图（多天叠加对比）

        参数:
            sensor_idx: 传感器索引
            days: 要显示的天数
        """
        plt.figure(figsize=(12, 7))

        for day in range(days):
            start_idx = day * 288
            end_idx = (day + 1) * 288

            if end_idx > len(self.time_index):
                break

            # 提取一天的数据
            day_data = self.data[start_idx:end_idx, sensor_idx]
            time_of_day = np.arange(288) * 5 / 60  # 转换为小时

            plt.plot(time_of_day, day_data,
                     label=f'第{day + 1}天',
                     linewidth=1.2,
                     alpha=0.7)

        plt.title(f'传感器 {sensor_idx} 的日流量模式')
        plt.xlabel('时间（小时）')
        plt.ylabel('流量（辆/5分钟）')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(np.arange(0, 25, 2))
        plt.tight_layout()
        plt.show()

    def plot_weekly_pattern(self, sensor_idx=0):
        """
        绘制周流量模式图（按星期几分组）

        参数:
            sensor_idx: 传感器索引
        """
        # 按星期几分组
        week_data = {i: [] for i in range(7)}  # 0-6对应周一到周日

        for i, timestamp in enumerate(self.time_index):
            day_of_week = timestamp.weekday()
            week_data[day_of_week].append(self.data[i, sensor_idx])

        # 计算每天的平均流量模式
        avg_week_pattern = np.zeros((7, 288))  # 7天，每天288个时间步

        for day in range(7):
            day_data = np.array(week_data[day])
            # 重塑为(天数, 每天时间步数)
            num_days = day_data.shape[0] // 288
            if num_days > 0:
                day_data = day_data[:num_days * 288].reshape(num_days, 288)
                avg_week_pattern[day] = np.mean(day_data, axis=0)

        # 绘制
        plt.figure(figsize=(14, 8))
        days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        time_of_day = np.arange(288) * 5 / 60  # 转换为小时

        for day in range(7):
            plt.plot(time_of_day, avg_week_pattern[day],
                     label=days[day],
                     linewidth=1.5)

        plt.title(f'传感器 {sensor_idx} 的周流量模式')
        plt.xlabel('时间（小时）')
        plt.ylabel('平均流量（辆/5分钟）')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(np.arange(0, 25, 2))
        plt.tight_layout()
        plt.show()

    def plot_flow_distribution(self, sensor_indices=None, bins=50):
        """
        绘制流量分布直方图

        参数:
            sensor_indices: 要包含的传感器索引列表，为None时使用所有传感器
            bins: 直方图的箱数
        """
        if sensor_indices is None:
            # 随机选择10个传感器
            sensor_indices = np.random.choice(self.data.shape[1], 10, replace=False)

        plt.figure(figsize=(12, 7))

        for sensor_idx in sensor_indices:
            sns.histplot(self.data[:, sensor_idx], bins=bins, kde=True,
                         label=f'传感器 {sensor_idx}', alpha=0.5)

        plt.title('交通流量分布')
        plt.xlabel('流量（辆/5分钟）')
        plt.ylabel('频率')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, sensor_indices=None, n_sensors=20):
        """
        绘制传感器间流量相关性矩阵

        参数:
            sensor_indices: 要包含的传感器索引列表，为None时随机选择
            n_sensors: 要显示的传感器数量
        """
        if sensor_indices is None:
            # 随机选择n_sensors个传感器
            sensor_indices = np.random.choice(self.data.shape[1], n_sensors, replace=False)

        # 计算相关性矩阵
        corr_matrix = np.zeros((n_sensors, n_sensors))
        for i in range(n_sensors):
            for j in range(n_sensors):
                if i <= j:
                    corr, _ = pearsonr(self.data[:, sensor_indices[i]],
                                       self.data[:, sensor_indices[j]])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=sensor_indices, yticklabels=sensor_indices)
        plt.title('传感器间流量相关性矩阵')
        plt.tight_layout()
        plt.show()

    def plot_sensor_heatmap(self, time_range=None, n_sensors=50):
        """
        绘制传感器流量热力图（时间×传感器）

        参数:
            time_range: 时间范围元组 (start_idx, end_idx)，为None时使用前288个时间步
            n_sensors: 要显示的传感器数量
        """
        if time_range is None:
            time_range = (0, min(288, len(self.time_index)))

        # 随机选择n_sensors个传感器
        sensor_indices = np.random.choice(self.data.shape[1], n_sensors, replace=False)

        # 提取数据
        heatmap_data = self.data[time_range[0]:time_range[1], sensor_indices].T

        # 标准化数据以便更好地显示
        scaler = StandardScaler()
        heatmap_data_scaled = scaler.fit_transform(heatmap_data)

        # 绘制热力图
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data_scaled, cmap="viridis",
                    xticklabels=False, yticklabels=sensor_indices)
        plt.title('传感器流量热力图（标准化后）')
        plt.xlabel('时间')
        plt.ylabel('传感器索引')
        plt.tight_layout()
        plt.show()

    def plot_traffic_heatmap_over_time(self, sensor_groups=None, days=7):
        """
        绘制多日流量热力图（按时间段分组）

        参数:
            sensor_groups: 传感器分组字典，例如 {'北部区域': [0,1,2], '南部区域': [3,4,5]}
            days: 要显示的天数
        """
        if sensor_groups is None:
            # 默认将传感器分为4组
            n_sensors = self.data.shape[1]
            step = n_sensors // 4
            sensor_groups = {
                '区域1': list(range(0, step)),
                '区域2': list(range(step, 2 * step)),
                '区域3': list(range(2 * step, 3 * step)),
                '区域4': list(range(3 * step, n_sensors))
            }

        time_steps = min(288 * days, len(self.time_index))

        # 计算每组的平均流量
        group_data = {}
        for group_name, indices in sensor_groups.items():
            group_data[group_name] = np.mean(self.data[:time_steps, indices], axis=1)

        # 重塑为(天数, 每天时间步数)
        num_days = time_steps // 288
        heatmap_data = np.zeros((len(sensor_groups), num_days, 288))

        for i, (group_name, data) in enumerate(group_data.items()):
            # 确保数据长度是288的整数倍
            valid_length = num_days * 288
            data = data[:valid_length].reshape(num_days, 288)
            heatmap_data[i] = data

        # 绘制热力图
        fig, axes = plt.subplots(len(sensor_groups), 1, figsize=(15, 4 * len(sensor_groups)))

        if len(sensor_groups) == 1:
            axes = [axes]

        for i, (group_name, data) in enumerate(group_data.items()):
            sns.heatmap(data, cmap="YlOrRd", ax=axes[i])
            axes[i].set_title(f'{group_name} 流量热力图')
            axes[i].set_xlabel('一天中的时间（5分钟间隔）')
            axes[i].set_ylabel('日期')
            axes[i].set_yticklabels([f'第{d + 1}天' for d in range(num_days)])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 创建可视化器实例
    # 如果有实际数据，可传入数据路径：visualizer = PEMS03Visualizer("path/to/pems03.npy")
    visualizer = PEMS03Visualizer(sample_rate=2)  # 采样率为2，减少数据量

    # 绘制时间序列图
    visualizer.plot_time_series(sensor_indices=[0, 50, 100, 150], days=3)

    # 绘制日流量模式图
    visualizer.plot_daily_pattern(sensor_idx=50, days=7)

    # 绘制周流量模式图
    visualizer.plot_weekly_pattern(sensor_idx=50)

    # 绘制流量分布直方图
    visualizer.plot_flow_distribution()

    # 绘制相关性矩阵
    visualizer.plot_correlation_matrix(n_sensors=15)

    # 绘制传感器流量热力图
    visualizer.plot_sensor_heatmap(time_range=(0, 288 * 3), n_sensors=40)

    # 绘制多日流量热力图
    visualizer.plot_traffic_heatmap_over_time(days=5)