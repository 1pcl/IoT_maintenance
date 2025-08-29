# -*- coding: utf-8 -*-
import math
import random
from utilities import DEFAULT_PARAM_VALUES
import matplotlib.pyplot as plt
import os

class SceneGenerator:
    @staticmethod
    def get_electricity_meter_config():
        DEFAULT_PARAM_VALUES["frequency_heartbeat"]=1800/60
        """电表场景配置"""
        return {
            "sensor_num": 100,
            "area_size": 500,
            "packet_size": 1000,
            "per_packet_cost": 5,
            "per_sensor_cost": 8,
            "budget": 500000,
            "development_cycle": 3,
            "life_span": 60,
            "per_wai_cost": 3,
            "installation_per_cost": 15,
            "frequency_sampling": 1800,
            "per_check_cost": 500,
            "cluster_head_percentage": 0.15,
            "max_hops": 4,
            "frames_per_round": 6,
            "application": "electricity_meter"
        }
    
    @staticmethod
    def get_animal_room_config():
        DEFAULT_PARAM_VALUES["frequency_heartbeat"]=900/60
        """动物房场景配置"""
        return {
            "sensor_num": 50,
            "area_size": 500,
            "packet_size": 3000,
            "per_packet_cost": 500,
            "per_sensor_cost": 20,
            "budget": 500000,
            "development_cycle": 8,
            "life_span": 24,
            "per_wai_cost": 8,
            "installation_per_cost": 5,
            "frequency_sampling": 900,
            "per_check_cost": 200,
            "cluster_head_percentage": 0.18,
            "max_hops": 2,
            "frames_per_round": 4,
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
            
            # 能耗参数
            "energy_sampling": 5e-9,
            "energy_consumption_tx": 5e-8,
            "energy_amplifier": 1e-10,
            "energy_consumption_rx": 5e-8,
            "energy_aggregation": 5e-9,
            "initial_energy": 5000,
            
            # 成本参数
            "development_people": 5,
            "monthly_salary": 10000,
            "per_power_cost": 5,
            "per_activation_cost": 2,
            "maintenance_instruction_size": 1,
            "maintenance_noise_size": 20,
            
            # 心跳包参数
            "heartbeat_packet_size": 2,
            
            # RTS/CTS参数
            "rts_cts_size": 20
        }
    
    @staticmethod
    def generate_nodes(config):
        """根据配置生成节点位置"""
        nodes = []
        application = config.get("application")
        sensor_num = config["sensor_num"]
        area_size = config["area_size"]
        
        for i in range(1, sensor_num + 1):
            if application == "electricity_meter":
                building_width = area_size * 0.8
                building_height = area_size * 2.6
                floor_height = building_height / 26
                center_x = area_size / 2
                floor = (i - 1) // 4 + 1
                unit = (i - 1) % 4
                y = floor * floor_height + 25
                
                if unit == 0:
                    x = center_x - building_width * 0.35
                elif unit == 1:
                    x = center_x - building_width * 0.25
                elif unit == 2:
                    x = center_x + building_width * 0.25
                else:
                    x = center_x + building_width * 0.35
                    
            elif application == "animal_room":
                i_chu = math.floor((i - 1) / 5)
                i_mode = (i - 1) % 5
                i_dex = (i - 5) % 10
                
                if i_chu == 0 or i_chu == 9:
                    if i_chu == 0:
                        x = 0
                        y = 50 + 100 * i_mode
                    else:
                        x = 500
                        y = 50 + 100 * i_mode
                else:
                    if i_chu == 1 or i_chu == 2:
                        x = 80
                    elif i_chu == 3 or i_chu == 4:
                        x = 80 + 60
                    elif i_chu == 5 or i_chu == 6:
                        x = 80 + 60 + 60 + 100 + 60
                    else:
                        x = 80 + 60 + 60 + 100 + 60 + 60
                        
                    if i_dex == 1:
                        y = 0
                    elif i_dex == 2:
                        y = 90
                    elif i_dex == 3:
                        y = 110
                    elif i_dex == 4:
                        y = 190
                    elif i_dex == 5:
                        y = 210
                    elif i_dex == 6:
                        y = 290
                    elif i_dex == 7:
                        y = 310
                    elif i_dex == 8:
                        y = 390
                    elif i_dex == 9:
                        y = 410
                    else:
                        y = 490
            else:
                # 默认随机分布
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
            x = (area_size / 4) * 3
            y = area_size + 25.0
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
            building_height = area_size * 2.6
            building_left = (area_size - building_width) / 2
            building_bottom = 0
            floor_height = building_height / 26
            
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
            elevator_left = center_x - building_width * 0.15
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
                elevator_left + elevator_width/2, 
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
            stairs_left = center_x + building_width * 0.05
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
                stairs_left + stairs_width/2, 
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
            # 绘制房间分隔线
            v_lines = [0, 200, 300, 500]
            for x in v_lines:
                plt.plot([x, x], [0, 500], color='blue', linestyle='-', linewidth=1)
            
            h_lines = [0, 100, 200, 300, 400, 500]
            for y in h_lines:
                plt.plot([0, 500], [y, y], color='blue', linestyle='-', linewidth=1)
            
            # 填充走廊区域
            plt.fill_between([200, 300], 0, 500, color='lightgray', alpha=0.3)
            
            # 添加走廊标签
            plt.text(250, 300, 'CORRIDOR', 
                    rotation=90, 
                    ha='center', va='center', 
                    fontsize=14, fontweight='bold',
                    color='darkgray')
        
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
        
        # # 添加建筑元素图例
        # if application == "electricity_meter":
        #     legend_elements.extend([
        #         plt.Line2D([0], [0], color='#F5F5F5', lw=4, label='Building (26 Floors)'),
        #         plt.Line2D([0], [0], color='#E8E8E8', lw=4, label='Elevator/Stairs'),
        #         plt.Line2D([0], [0], color='#F0F0FF', lw=4, label='Lobby'),
        #         plt.Line2D([0], [0], color='#8B4513', lw=4, label='Main Entrance')
        #     ])
        # elif application == "animal_room":
        #     legend_elements.append(plt.Line2D([0], [0], color='blue', lw=2, label='Room Walls'))
        #     legend_elements.append(plt.Line2D([0], [0], color='lightgray', lw=4, label='Corridor'))

        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        # 设置图例字体大小
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
        
        # 设置刻度字体大小
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        # 设置坐标轴范围
        if application == "electricity_meter":
            plt.xlim([building_left - 10, building_left + building_width + 30])
            plt.ylim([-building_height*0.05, building_height*1.05])
        elif application == "animal_room":
            plt.xlim([-10, 550])
            plt.ylim([-10, 550])
            
        plt.grid(True, alpha=0.2)
        
        # 保存图片
        save_path = os.path.join(save_folder, f'initial_topology_{application}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
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