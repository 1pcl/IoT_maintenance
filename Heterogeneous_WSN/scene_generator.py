# -*- coding: utf-8 -*-
import math
import random
import matplotlib.pyplot as plt
import os
import numpy as np

class SceneGenerator:
    @staticmethod
    def get_four_zone_config():
        """四个区域场景配置"""        
        # 基础配置
        base_config = {
            "sensor_num": 400,  # 总节点数
            "area_size": 500,   # 正方形区域大小
            "budget": 1500000,
            "development_cycle": 6,
            "life_span": 24,
            "development_people": 8,
            "monthly_salary": 25000,           
            # LEACH协议参数
            "intra_cluster_multihop": 0,
            "inter_cluster_multihop": 1,
            "configuring_package_size": 15,
            "TDMA_package_size": 250,
            # RTS/CTS参数
            "rts_cts_size": 25,
            "application": "four_zone"
        }
        
        # 四个区域的特定参数 - 基于实际应用场景
        zone_configs = {
            "zone_1": {  # 区域1: 工业监控区 - 高功耗传感器
                "packet_size": 2000,  # 大数据包，包含详细监控数据
                "per_packet_cost": 8,
                "per_sensor_cost": 80,  # 工业级传感器更昂贵
                "per_wai_cost": 12,
                "installation_per_cost": 35,
                "frequency_sampling": 60,  # 高频采样，实时监控
                "per_check_cost": 800,
                "cluster_head_percentage": 0.15,
                "max_hops": 3,
                "frames_per_round": 6,
                "color": "red",
                "description": "Industrial Monitoring",                
                # 能耗参数 - 工业级高功耗传感器
                "energy_sampling": 5e-9,      # 高采样能耗
                "energy_consumption_tx": 8e-8, # 高传输功率
                "energy_amplifier": 1.2e-10,   # 高放大系数
                "energy_consumption_rx": 6e-8, # 高接收功耗
                "energy_aggregation": 6e-9,    # 复杂数据处理
                "initial_energy": 15000,       # 更大电池容量
                # 成本参数
                "per_power_cost": 10,
                "per_activation_cost": 5,
                "maintenance_instruction_size": 8,
                "maintenance_noise_size": 50,                
                # 心跳包参数
                "heartbeat_packet_size": 20,       
                "sensor_num": 0,        # 初始化为0，后续更新
            },
            "zone_2": {  # 区域2: 环境监测区 - 中等功耗
                "packet_size": 800,
                "per_packet_cost": 5,
                "per_sensor_cost": 45,
                "per_wai_cost": 8,
                "installation_per_cost": 25,
                "frequency_sampling": 300,  # 中等采样频率
                "per_check_cost": 500,
                "cluster_head_percentage": 0.12,
                "max_hops": 4,
                "frames_per_round": 8,
                "color": "blue",
                "description": "Environmental Monitoring",                
                # 能耗参数 - 环境传感器中等功耗
                "energy_sampling": 3e-9,
                "energy_consumption_tx": 5e-8,
                "energy_amplifier": 9e-11,
                "energy_consumption_rx": 4.5e-8,
                "energy_aggregation": 4e-9,
                "initial_energy": 10000,
                # 成本参数
                "per_power_cost": 6,
                "per_activation_cost": 3,
                "maintenance_instruction_size": 5,
                "maintenance_noise_size": 30,               
                # 心跳包参数
                "heartbeat_packet_size": 15,     
                "sensor_num": 0,           # 初始化为0，后续更新
            },
            "zone_3": {  # 区域3: 安防监控区 - 低功耗但需要长距离通信
                "packet_size": 1500,  # 中等数据包，包含图像/状态信息
                "per_packet_cost": 7,
                "per_sensor_cost": 60,
                "per_wai_cost": 10,
                "installation_per_cost": 30,
                "frequency_sampling": 900,  # 较低采样频率
                "per_check_cost": 600,
                "cluster_head_percentage": 0.18,  # 更多簇头保证可靠性
                "max_hops": 2,  # 较少跳数保证实时性
                "frames_per_round": 10,
                "color": "green",
                "description": "Security Monitoring",                
                # 能耗参数 - 安防传感器特殊需求
                "energy_sampling": 2e-9,      # 低采样能耗
                "energy_consumption_tx": 7e-8, # 高传输功率用于长距离
                "energy_amplifier": 1.5e-10,   # 高放大用于覆盖范围
                "energy_consumption_rx": 3e-8, # 中等接收功耗
                "energy_aggregation": 3e-9,
                "initial_energy": 12000,
                # 成本参数
                "per_power_cost": 5,
                "per_activation_cost": 2,
                "maintenance_instruction_size": 3,
                "maintenance_noise_size": 25,               
                # 心跳包参数
                "heartbeat_packet_size": 8, 
                "sensor_num": 0,  # 初始化为0，后续更新
            },
            "zone_4": {  # 区域4: 基础设施监控 - 超低功耗
                "packet_size": 400,  # 小数据包，状态信息
                "per_packet_cost": 3,
                "per_sensor_cost": 35,
                "per_wai_cost": 6,
                "installation_per_cost": 20,
                "frequency_sampling": 1800,  # 低频采样，节能
                "per_check_cost": 300,
                "cluster_head_percentage": 0.08,  # 较少簇头
                "max_hops": 5,  # 可接受多跳
                "frames_per_round": 12,
                "color": "orange",
                "description": "Infrastructure Monitoring",                
                # 能耗参数 - 超低功耗设计
                "energy_sampling": 1e-9,      # 极低采样能耗
                "energy_consumption_tx": 3e-8, # 低传输功率
                "energy_amplifier": 6e-11,     # 低放大系数
                "energy_consumption_rx": 2e-8, # 低接收功耗
                "energy_aggregation": 2e-9,    # 简单数据处理
                "initial_energy": 8000,        # 较小电池
                # 成本参数
                "per_power_cost": 2,
                "per_activation_cost": 1.5,
                "maintenance_instruction_size": 2,
                "maintenance_noise_size": 15,               
                # 心跳包参数
                "heartbeat_packet_size": 4, 
                "sensor_num": 0,  # 初始化为0，后续更新
            }
        }
        
        return base_config, zone_configs
    
    @staticmethod
    def generate_nodes(config, zone_configs):
        """根据配置生成节点位置，并分配到四个区域"""
        nodes = []
        sensor_num = config["sensor_num"]
        area_size = config["area_size"]
        half_size = area_size / 2
        
        # 计算每个区域的节点数（均匀分布）
        nodes_per_zone = sensor_num // 4
        
        # 区域边界
        zones = {
            "zone_1": {"x_range": (0, half_size), "y_range": (half_size, area_size)},      # 左上
            "zone_2": {"x_range": (half_size, area_size), "y_range": (half_size, area_size)}, # 右上
            "zone_3": {"x_range": (half_size, area_size), "y_range": (0, half_size)},      # 右下
            "zone_4": {"x_range": (0, half_size), "y_range": (0, half_size)}               # 左下
        }
        
        # 初始化每个区域的节点计数器
        zone_node_counts = {zone_name: 0 for zone_name in zones.keys()}
        
        node_id = 1
        for zone_name, zone_info in zones.items():
            zone_params = zone_configs[zone_name]
            x_range = zone_info["x_range"]
            y_range = zone_info["y_range"]
            
            for i in range(nodes_per_zone):
                x = round(random.uniform(x_range[0], x_range[1]))
                y = round(random.uniform(y_range[0], y_range[1]))
                
                nodes.append({
                    "id": node_id,
                    "x": x,
                    "y": y,
                    "zone": zone_name,
                    "zone_params": zone_params.copy()  # 存储该区域的特定参数
                })
                node_id += 1
                zone_node_counts[zone_name] += 1
        
        # 如果节点数不能被4整除，将剩余节点随机分配到区域中
        remaining_nodes = sensor_num - nodes_per_zone * 4
        for i in range(remaining_nodes):
            zone_name = random.choice(list(zones.keys()))
            zone_info = zones[zone_name]
            zone_params = zone_configs[zone_name]
            
            x = round(random.uniform(zone_info["x_range"][0], zone_info["x_range"][1]))
            y = round(random.uniform(zone_info["y_range"][0], zone_info["y_range"][1]))
            
            nodes.append({
                "id": node_id,
                "x": x,
                "y": y,
                "zone": zone_name,
                "zone_params": zone_params.copy()
            })
            node_id += 1
            zone_node_counts[zone_name] += 1
            
        return nodes, zone_node_counts
    
    @staticmethod
    def generate_base_station(config):
        """生成基站位置 - 放在区域中心"""
        area_size = config["area_size"]
        
        return {
            "id": 0,
            "x": area_size / 2,
            "y": area_size / 2,
            "distance": 0,
            "section": 0
        }
    
    @staticmethod
    def plot_initial_topology(config, zone_configs, nodes, base_station, save_folder="visualizations"):
        """
        绘制初始拓扑图，显示四个区域和能耗特性
        """
        # 创建保存目录
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # 创建图形
        plt.figure(figsize=(14, 10))
        
        area_size = config["area_size"]
        half_size = area_size / 2
        
        # 绘制区域分割线
        plt.axvline(x=half_size, color='black', linestyle='--', linewidth=2, alpha=0.7)
        plt.axhline(y=half_size, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        # 绘制区域背景和标签
        zones = {
            "zone_1": {"x_range": (0, half_size), "y_range": (half_size, area_size)},
            "zone_2": {"x_range": (half_size, area_size), "y_range": (half_size, area_size)},
            "zone_3": {"x_range": (half_size, area_size), "y_range": (0, half_size)},
            "zone_4": {"x_range": (0, half_size), "y_range": (0, half_size)}
        }
        
        # 为每个区域添加半透明背景和标签
        for zone_name, zone_info in zones.items():
            zone_params = zone_configs[zone_name]
            x_range = zone_info["x_range"]
            y_range = zone_info["y_range"]
            
            # 添加半透明背景
            rect = plt.Rectangle(
                (x_range[0], y_range[0]),
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                fill=True,
                color=zone_params["color"],
                alpha=0.1
            )
            plt.gca().add_patch(rect)
            
        
        # 按区域绘制节点
        zone_colors = {zone: config["color"] for zone, config in zone_configs.items()}
        zone_nodes = {zone: [] for zone in zone_configs.keys()}
        
        for node in nodes:
            zone_nodes[node["zone"]].append(node)
        
        # 绘制每个区域的节点
        legend_elements = []
        for zone_name, zone_nodes_list in zone_nodes.items():
            if zone_nodes_list:
                color = zone_colors[zone_name]
                x_coords = [node["x"] for node in zone_nodes_list]
                y_coords = [node["y"] for node in zone_nodes_list]
                
                plt.scatter(x_coords, y_coords, color=color, s=50, alpha=0.7, label=f'Zone {zone_name[-1]}')
                
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                             markersize=8, label=f'Zone {zone_name[-1]} ({len(zone_nodes_list)} nodes)')
                )
        
        # 绘制基站
        plt.plot(base_station["x"], base_station["y"], 's', 
                color='purple', markersize=20, label='Base Station', markeredgecolor='black', markeredgewidth=2)
        
        legend_elements.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', 
                     markersize=12, label='Base Station', markeredgecolor='black', markeredgewidth=1)
        )
        
        # 设置图例
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
        
        # 设置坐标轴
        plt.xlim(0, area_size)
        plt.ylim(0, area_size)
        plt.grid(True, alpha=0.3)       
        # 设置刻度字体大小
        plt.tick_params(axis='both', which='major', labelsize=16) 
        
        # 保存图片
        save_path = os.path.join(save_folder, 'initial_topology_four_zone_energy.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    @staticmethod
    def create_scene(visualize=True):
        """创建四区域场景"""
        config, zone_configs = SceneGenerator.get_four_zone_config()
        
        # 生成节点和基站
        nodes, zone_node_counts = SceneGenerator.generate_nodes(config, zone_configs)
        base_station = SceneGenerator.generate_base_station(config)
        
        # 更新每个区域的传感器节点数量
        for zone_name, count in zone_node_counts.items():
            zone_configs[zone_name]["sensor_num"] = count
        
        # 绘制初始拓扑图和能耗对比图
        if visualize:
            SceneGenerator.plot_initial_topology(config, zone_configs, nodes, base_station)
        
        return {
            "config": config,
            "zone_configs": zone_configs,
            "nodes": nodes,
            "base_station": base_station
        }

# 使用示例
if __name__ == "__main__":
    # 创建四区域场景
    scene = SceneGenerator.create_scene(visualize=True)
    
    # 打印每个区域的传感器节点数量
    print("各区域传感器节点数量:")
    for zone_name, zone_config in scene["zone_configs"].items():
        print(f"{zone_name}: {zone_config['sensor_num']} 个传感器节点")
    
    print(f"\n总传感器节点数: {sum(zone_config['sensor_num'] for zone_config in scene['zone_configs'].values())}")