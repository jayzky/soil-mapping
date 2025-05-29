import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple

class GridGenerator:
    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float = 0.1):
        """
        初始化网格生成器
        
        Args:
            bounds: 边界范围 (min_lat, max_lat, min_lon, max_lon)
            resolution: 网格分辨率（度）
        """
        self.bounds = bounds  # (min_lat, max_lat, min_lon, max_lon)
        self.resolution = resolution
        
        # 湖北省主要城市及其大致经纬度
        self.cities = {
            '武汉': (30.5928, 114.3055),
            '黄石': (30.2147, 115.0382),
            '十堰': (32.6475, 110.7871),
            '宜昌': (30.7000, 111.2800),
            '襄阳': (32.0090, 112.1220),
            '鄂州': (30.3875, 114.8947),
            '荆门': (31.0354, 112.1990),
            '孝感': (30.9240, 113.9170),
            '荆州': (30.3340, 112.2410),
            '黄冈': (30.4461, 114.8720),
            '咸宁': (29.8410, 114.3220),
            '随州': (31.6900, 113.3830),
            '恩施': (30.2720, 109.4880),
            '仙桃': (30.3620, 113.4540),
            '天门': (30.6630, 113.1660),
            '潜江': (30.4021, 112.8990)
        }
        
    def generate_grid(self) -> pd.DataFrame:
        """生成网格点"""
        min_lat, max_lat, min_lon, max_lon = self.bounds
        lats = np.arange(min_lat, max_lat, self.resolution)
        lons = np.arange(min_lon, max_lon, self.resolution)
        
        # 使用meshgrid生成网格点
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        
        # 创建DataFrame
        grid_df = pd.DataFrame({
            'latitude': lat_grid.flatten(),
            'longitude': lon_grid.flatten()
        })
        
        # 为每个网格点分配最近的城市
        grid_df['city'] = grid_df.apply(self._assign_nearest_city, axis=1)
        
        return grid_df
    
    def _assign_nearest_city(self, row):
        """为给定的经纬度找到最近的城市"""
        min_dist = float('inf')
        nearest_city = None
        
        for city, (city_lat, city_lon) in self.cities.items():
            dist = np.sqrt((row['latitude'] - city_lat)**2 + (row['longitude'] - city_lon)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_city = city
                
        return nearest_city
    
    def generate_features(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """
        为网格点生成特征
        
        Args:
            grid_df: 包含经纬度的DataFrame
        
        Returns:
            包含所有特征的DataFrame
        """
        features = pd.DataFrame()
        
        # 生成高程数据（基于经纬度的简单模型）
        lat_factor = (grid_df['latitude'] - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        lon_factor = (grid_df['longitude'] - self.bounds[2]) / (self.bounds[3] - self.bounds[2])
        
        features['elevation'] = 100 + 900 * lat_factor + \
                              200 * np.sin(2 * np.pi * lon_factor) + \
                              np.random.normal(0, 20, len(grid_df))
        features['elevation'] = np.clip(features['elevation'], 20, 1200)
        
        # 生成坡度数据（与高程相关）
        features['slope'] = 5 + 25 * lat_factor + \
                          10 * np.sin(2 * np.pi * lon_factor) + \
                          np.random.normal(0, 2, len(grid_df))
        features['slope'] = np.clip(features['slope'], 0, 30)
        
        # 生成坡向数据
        features['aspect'] = 180 + 180 * np.sin(2 * np.pi * lat_factor) + \
                           90 * np.cos(2 * np.pi * lon_factor) + \
                           np.random.normal(0, 10, len(grid_df))
        features['aspect'] = np.mod(features['aspect'], 360)
        
        # 生成NDVI数据（基于经纬度和高程的简单模型）
        features['ndvi'] = 0.5 + 0.3 * np.sin(grid_df['latitude'] * np.pi / 180) + \
                          0.2 * np.cos(grid_df['longitude'] * np.pi / 180) + \
                          0.1 * np.sin(features['elevation'] / 1200 * np.pi) + \
                          np.random.normal(0, 0.05, len(grid_df))
        features['ndvi'] = np.clip(features['ndvi'], 0, 1)
        
        # 生成降水数据（基于经纬度的简单模型）
        features['precipitation'] = 1000 + 500 * lat_factor - \
                                  200 * lon_factor + \
                                  np.random.normal(0, 50, len(grid_df))
        features['precipitation'] = np.clip(features['precipitation'], 800, 1500)
        
        # 生成温度数据（基于经纬度和高程的简单模型）
        features['temperature'] = 16 - 6 * lat_factor + \
                                2 * lon_factor - \
                                0.6 * (features['elevation'] / 100) + \
                                np.random.normal(0, 0.5, len(grid_df))
        features['temperature'] = np.clip(features['temperature'], 10, 25)
        
        return features 