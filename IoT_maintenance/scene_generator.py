# -*- coding: utf-8 -*-
import math
import random
from utilities import DEFAULT_PARAM_VALUES
import matplotlib.pyplot as plt
import os
import numpy as np

class SceneGenerator:
    @staticmethod
    def get_electricity_meter_config():
        DEFAULT_PARAM_VALUES["frequency_heartbeat"] = 1800/60
        """电表场景配置 - 扩展到约200个节点"""
        return {
            "sensor_num": 208,  # 26层 × 8户/层 = 208个节点
            "area_size": 500,
            "packet_size": 1500,  # 增加数据包大小，更符合实际电表数据
            "per_packet_cost": 5,  # 降低单包成本，符合实际通信成本
            "per_sensor_cost": 25,  # 增加单个传感器成本，符合高质量电表价格
            "budget": 1000000,  # 增加预算，符合实际项目规模
            "development_cycle": 4,  # 增加开发周期，符合实际项目时间
            "life_span": 60,  # 生命周期
            "per_wai_cost": 5,  # 调整维护成本
            "installation_per_cost": 30,  # 增加安装成本，符合人工费用
            "frequency_sampling": 1800,
            "per_check_cost": 800,  # 增加检查成本
            "cluster_head_percentage": 0.12,  # 调整簇头比例
            "max_hops": 5,  # 增加最大跳数
            "frames_per_round": 8,  # 增加每轮帧数
            "application": "electricity_meter"
        }    
    
    @staticmethod
    def get_animal_room_config():
        DEFAULT_PARAM_VALUES["frequency_heartbeat"] = 900/60
        """动物房场景配置 - 扩展到约300个节点"""
        return {
            "sensor_num": 300,  # 30列 × 10个/列 = 300个节点
            "area_size": 800,  # 增加区域大小，容纳更多节点
            "packet_size": 5000,  # 增加数据包大小，动物监测需要更多数据
            "per_packet_cost": 50,  # 调整单包成本
            "per_sensor_cost": 45,  # 增加传感器成本，动物监测传感器更昂贵
            "budget": 1500000,  # 增加预算
            "development_cycle": 6,  # 调整开发周期
            "life_span": 24,  # 生命周期
            "per_wai_cost": 12,  # 增加维护成本
            "installation_per_cost": 20,  # 调整安装成本
            "frequency_sampling": 600,  # 增加采样频率，动物监测需要更频繁
            "per_check_cost": 300,  # 调整检查成本
            "cluster_head_percentage": 0.15,  # 调整簇头比例
            "max_hops": 3,  # 调整最大跳数
            "frames_per_round": 6,  # 调整每轮帧数
            "application": "animal_room"
        }
        
    @staticmethod
    def get_base_config():
        """基础配置（不随场景变化的部分）"""
        return {
            # LEACH协议参数
            "intra_cluster_multihop": 0,
            "inter_cluster_multihop": 1,
            "configuring_package_size": 15,
            "TDMA_package_size": 250,
            
            # 能耗参数 - 更新为更实际的能耗值
            "energy_sampling": 3e-9,
            "energy_consumption_tx": 4.5e-8,
            "energy_amplifier": 8e-11,
            "energy_consumption_rx": 4.5e-8,
            "energy_aggregation": 4e-9,
            "initial_energy": 10000,  # 增加初始能量
            
            # 成本参数 - 更新为更实际的市场值
            "development_people": 8,  # 增加开发人员
            "monthly_salary": 25000,  # 增加月薪
            "per_power_cost": 0.8,  # 调整电力成本
            "per_activation_cost": 1.5,  # 调整激活成本
            "maintenance_instruction_size": 2,
            "maintenance_noise_size": 25,
            
            # 心跳包参数
            "heartbeat_packet_size": 3,  # 增加心跳包大小
            
            # RTS/CTS参数
            "rts_cts_size": 25  # 增加RTS/CTS大小
        }
    
    @staticmethod
    def generate_nodes(config):
        """根据配置生成节点位置"""
        nodes = []
        application = config.get("application")
        sensor_num = config["sensor_num"]
        area_size = config["area_size"]
        
        if application == "electricity_meter":
            # 电表场景 - 26层建筑，每层8户（4户左侧，4户右侧）
            building_width = area_size * 0.8
            building_height = area_size * 2.6
            floor_height = building_height / 26
            center_x = area_size / 2
            
            for i in range(1, sensor_num + 1):
                floor = (i - 1) // 8 + 1  # 每层8户
                unit = (i - 1) % 8
                y = floor * floor_height + 25
                
                # 左侧4户
                if unit < 4:
                    x = center_x - building_width * (0.4 - unit * 0.1)
                # 右侧4户
                else:
                    x = center_x + building_width * (0.1 + (unit - 4) * 0.1)
                    
                nodes.append({
                    "id": i,
                    "x": x,
                    "y": y
                })
                
        elif application == "animal_room":
            # 动物房场景 - 30列，每列10个节点
            # 走廊在x=380到420之间 (宽度40，更加合理)
            corridor_left = 380
            corridor_right = 420
            
            # 左侧房间列数 (0-380)
            left_columns = 15
            # 右侧房间列数 (420-800)
            right_columns = 15
            
            # 左侧列x坐标
            left_x_positions = np.linspace(20, corridor_left-20, left_columns)
            # 右侧列x坐标
            right_x_positions = np.linspace(corridor_right+20, area_size-20, right_columns)
            
            # 所有列的x坐标
            all_x_positions = np.concatenate([left_x_positions, right_x_positions])
            
            # 每列的y坐标 (10个均匀分布的点)
            y_positions = np.linspace(40, area_size-40, 10)
            
            for i in range(1, sensor_num + 1):
                col_idx = (i - 1) // 10
                row_idx = (i - 1) % 10
                
                x = all_x_positions[col_idx]
                y = y_positions[row_idx]
                
                nodes.append({
                    "id": i,
                    "x": x,
                    "y": y
                })
        else:
            # 默认随机分布
            for i in range(1, sensor_num + 1):
                x = round(random.uniform(0, area_size))
                y = round(random.uniform(0, area_size))
                
                nodes.append({
                    "id": i,
                    "x": x,
                    "y": y
                })
            
        return nodes
    
    @staticmethod
    def generate_base_station(config):
        """根据配置生成基站位置"""
        application = config.get("application", "default")
        area_size = config["area_size"]
        
        if application == "electricity_meter":
            building_width = area_size * 0.8
            center_x = area_size / 2
            building_height = area_size * 2.6
            lobby_height = building_height / 26
            x = center_x - building_width * 0.25
            y = lobby_height * 0.5
        elif application == "animal_room":
            # 基站位于走廊中央上方
            corridor_left = 380
            corridor_right = 420
            x = (corridor_left + corridor_right) / 2  # 走廊中央
            y = area_size + 50.0  # 区域上方
        else:
            x = area_size + 25.0
            y = area_size / 2
            
        return {
            "id": 0,
            "x": x,
            "y": y,
            "distance": 0,
            "section": 0
        }
    
    @staticmethod
    def plot_initial_topology(config, nodes, base_station, save_folder="visualizations"):
        """
        绘制初始拓扑图
        """
        # 创建保存目录
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # 创建图形
        plt.figure(figsize=(12, 16))
        
        application = config.get("application", "default")
        area_size = config["area_size"]
        
        # 电表场景的特殊处理
        if application == "electricity_meter":
            # 建筑物参数
            building_width = area_size * 0.8
            building_height = area_size * 2.7
            building_left = (area_size - building_width) / 2
            building_bottom = 0
            floor_height = building_height / 27
            
            # 绘制建筑物轮廓
            building = plt.Rectangle(
                (building_left, building_bottom), 
                building_width, 
                building_height,
                fill=True, 
                color='#F5F5F5',
                edgecolor='#333333',
                linewidth=2
            )
            plt.gca().add_patch(building)
            
            # 绘制楼层分隔线
            for i in range(1, 26):
                y_pos = building_bottom + i * floor_height
                plt.plot(
                    [building_left, building_left + building_width],
                    [y_pos, y_pos],
                    color='#CCCCCC',
                    linewidth=0.7,
                    linestyle='-'
                )
            
            # 绘制电梯
            center_x = area_size / 2
            elevator_width = building_width * 0.1
            elevator_height = building_height - floor_height
            elevator_left = center_x - building_width * 0.05
            elevator_bottom = building_bottom + floor_height
            
            elevator = plt.Rectangle(
                (elevator_left, elevator_bottom),
                elevator_width,
                elevator_height,
                fill=True,
                color='#E8E8E8',
                edgecolor='#999999',
                linewidth=1.5
            )
            plt.gca().add_patch(elevator)
            
            # 添加"ELEVATOR"文字
            plt.text(
                elevator_left + elevator_width/4, 
                elevator_bottom + elevator_height/2,
                'ELEVATOR',
                rotation=90,
                ha='center',
                va='center',
                fontsize=14,
                fontweight='bold',
                color='#555555'
            )
            
            # 绘制楼梯
            stairs_width = building_width * 0.1
            stairs_height = building_height - floor_height
            stairs_left = center_x-building_width * 0.05
            stairs_bottom = building_bottom + floor_height
            
            stairs = plt.Rectangle(
                (stairs_left, stairs_bottom),
                stairs_width,
                stairs_height,
                fill=True,
                color='#E8E8E8',
                edgecolor='#999999',
                linewidth=1.5
            )
            plt.gca().add_patch(stairs)
            
            # 添加"STAIRS"文字
            plt.text(
                stairs_left+building_width * 0.05 + stairs_width/4, 
                stairs_bottom + stairs_height/2,
                'STAIRS',
                rotation=90,
                ha='center',
                va='center',
                fontsize=14,
                fontweight='bold',
                color='#555555'
            )
            
            # 绘制大堂区域
            lobby_height = floor_height
            lobby = plt.Rectangle(
                (building_left, building_bottom),
                building_width,
                lobby_height,
                fill=True,
                color='#F0F0FF',
                edgecolor='#333333',
                linewidth=1.5,
                alpha=0.7
            )
            plt.gca().add_patch(lobby)
            
            # 绘制大门
            door_width = building_width * 0.15
            door_height = lobby_height * 0.7
            door_left = center_x - door_width/2
            door_bottom = building_bottom
            
            door = plt.Rectangle(
                (door_left, door_bottom),
                door_width,
                door_height,
                fill=True,
                color='#8B4513',
                linewidth=1.5
            )
            plt.gca().add_patch(door)
            
            # 添加"MAIN ENTRANCE"标签
            plt.text(
                center_x,
                door_bottom + door_height + 5,
                'MAIN ENTRANCE',
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold',
                color='#8B4513'
            )
        
        # 动物房场景的特殊处理
        elif application == "animal_room":
            # 绘制走廊 (宽度40，更加合理)
            corridor_left = 380
            corridor_right = 420
            corridor = plt.Rectangle(
                (corridor_left, 0),
                corridor_right - corridor_left,
                area_size,
                fill=True,
                color='#F0F0F0',
                alpha=0.5
            )
            plt.gca().add_patch(corridor)
            
            # 添加走廊标签
            plt.text(
                (corridor_left + corridor_right) / 2,
                area_size / 2,
                'CORRIDOR',
                rotation=90,
                ha='center',
                va='center',
                fontsize=16,
                fontweight='bold',
                color='#666666'
            )
            
            # 绘制房间分隔线
            # 左侧房间分隔线
            left_room_width = (corridor_left) / 15
            for i in range(1, 15):
                x = left_room_width * i
                plt.plot([x, x], [0, area_size], 'gray', linestyle='-', linewidth=0.5, alpha=0.7)
            
            # 右侧房间分隔线
            right_room_width = (area_size - corridor_right) / 15
            for i in range(1, 15):
                x = corridor_right + right_room_width * i
                plt.plot([x, x], [0, area_size], 'gray', linestyle='-', linewidth=0.5, alpha=0.7)
            
            # 水平分隔线 (每列10个节点，所以10行)
            row_height = area_size / 10
            for i in range(1, 10):
                y = row_height * i
                plt.plot([0, area_size], [y, y], 'gray', linestyle='-', linewidth=0.5, alpha=0.7)
        
        # 绘制基站
        plt.plot(base_station["x"], base_station["y"], 's', 
                color='purple', markersize=18, label='Base Station')
        
        # 绘制所有节点
        for node in nodes:
            plt.plot(node["x"], node["y"], 'o', color='green', markersize=8, alpha=0.7)
        
        # 设置图例
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', 
                    markersize=12, label='Base Station'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                    markersize=8, label=f'Nodes: {len(nodes)}'),
        ]
        
        # 设置图例字体大小
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
        
        # 设置刻度字体大小
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        # 设置坐标轴范围
        if application == "electricity_meter":
            plt.xlim([building_left - 10, building_left + building_width + 30])
            plt.ylim([-building_height*0.05, building_height*1.05])
        elif application == "animal_room":
            plt.xlim([-10, area_size + 10])
            plt.ylim([-10, area_size + 60])  # 为基站留出空间
            
        plt.grid(True, alpha=0.2)
        plt.xlabel('X Coordinate (m)', fontsize=16)
        plt.ylabel('Y Coordinate (m)', fontsize=16)
        plt.title(f'Initial Topology - {application.replace("_", " ").title()}', fontsize=18)
        
        # 保存图片
        save_path = os.path.join(save_folder, f'initial_topology_{application}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    @staticmethod
    def create_scene(scene_type, visualize=True):
        """创建指定类型的场景"""
        if scene_type == "electricity_meter":
            config = SceneGenerator.get_electricity_meter_config()
        elif scene_type == "animal_room":
            config = SceneGenerator.get_animal_room_config()
        else:
            raise ValueError(f"未知的场景类型: {scene_type}")
        
        # 合并基础配置
        base_config = SceneGenerator.get_base_config()
        config.update(base_config)
        
        # 生成节点和基站
        nodes = SceneGenerator.generate_nodes(config)
        base_station = SceneGenerator.generate_base_station(config)
        
        # 绘制初始拓扑图
        if visualize:
            SceneGenerator.plot_initial_topology(config, nodes, base_station)
        
        return {
            "config": config,
            "nodes": nodes,
            "base_station": base_station
        }

