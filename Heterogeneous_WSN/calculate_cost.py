import random
from collections import defaultdict, deque
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import concurrent.futures
from deap import base, creator, tools, algorithms
import json
from PIL import Image  
from utilities import MODULES,FAULT_TYPES
from module_cost import set_modules_cost,calculate_hardware_cost,calculate_development_cost,calculate_installation_cost

# ======================== 成本计算函数 ========================

def calculate_total_cost_with_simulation(wsn_sim, selected_modules_by_zone,scene_data):
    """计算系统总成本"""
    config = scene_data["config"]
    # 从全局配置中获取参数
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    development_cycle = config["development_cycle"]
    # 计算开发成本
    developmentCost = calculate_development_cost(developmentPeople, monthly_salary, development_cycle)  # 开发成本

    zone_configs = scene_data["zone_configs"]

    total_installation_cost=0
    total_hardware_cost=0
    # 为每个区域单独计算成本
    for zone_name, zone_config in zone_configs.items():
        sensor_num = zone_config["sensor_num"]
        
        # 区域特定的成本参数
        installation_per_cost = zone_config["installation_per_cost"]
        sensor_cost = zone_config["per_sensor_cost"]        
        baseCost = sensor_cost * 1.5  # 基站成本
        # 计算该区域的各项成本
        zone_installation_cost = calculate_installation_cost(sensor_num, installation_per_cost) + calculate_hardware_cost(1, baseCost) 
        zone_hardware_cost = calculate_hardware_cost(sensor_num, sensor_cost)
        
        # 累加到总成本
        total_installation_cost += zone_installation_cost
        total_hardware_cost += zone_hardware_cost

    base_cost = developmentCost + total_installation_cost + total_hardware_cost
    # 设置模块成本 
    set_modules_cost(scene_data)
    
    # 维护模块成本 
    # module_cost = sum(MODULES[module]["cost"] for module in selected_modules)
    # 计算四个区域的模块成本
    module_costs_by_zone = {}

    for zone, modules in selected_modules_by_zone.items():
        total_cost = 0
        for module in modules:
            if module in MODULES[zone]:
                total_cost += MODULES[zone][module]["cost"]
        module_costs_by_zone[zone] = total_cost

    # 计算总成本
    total_module_cost = sum(module_costs_by_zone.values())
    
    # 获取仿真结果 
    # loss_data_count,node_fault,check_count=wsn_sim.run_simulation() 
    loss_data_count, node_fault_list, check_count = wsn_sim.run_simulation() 

    # 初始化总代价
    total_data_loss_cost = 0
    total_fault_cost = 0
    total_check_cost = 0

    # 按区域计算代价
    for zone_name in zone_configs.keys():
        zone_config = zone_configs[zone_name]
        
        # 获取区域特定的参数
        per_packet_cost = zone_config.get("per_packet_cost", 1.0)  # 默认值1.0
        per_check_cost = zone_config.get("per_check_cost", 5.0)    # 默认值5.0
        frames_per_round = zone_config["frames_per_round"]
        
        # 该区域的数据丢失包数
        zone_loss_data = loss_data_count.get(zone_name, 0)
        
        # 计算该区域的故障次数（所有节点所有故障类型发生次数的总和）
        zone_fault_count = 0
        if zone_name in node_fault_list:
            for node_fault in node_fault_list[zone_name]:
                for fault_type, timer in node_fault["fault_timers"].items():
                    zone_fault_count += timer["count"]
        
        # 数据丢失成本 = 数据包丢失成本 + 故障导致的数据包丢失成本
        zone_data_loss_cost = zone_loss_data * per_packet_cost + zone_fault_count * per_packet_cost * frames_per_round
        total_data_loss_cost += zone_data_loss_cost
        
        # 计算该区域的故障维修成本（按故障类型分类统计）
        zone_fault_cost = 0
        if zone_name in FAULT_TYPES:
            for fault_type, params in FAULT_TYPES[zone_name].items():
                # 统计该故障类型在该区域所有节点中的发生总次数
                type_count = 0
                if zone_name in node_fault_list:
                    for node_fault in node_fault_list[zone_name]:
                        if fault_type in node_fault["fault_timers"]:
                            type_count += node_fault["fault_timers"][fault_type]["count"]
                # 累加该故障类型的总成本
                zone_fault_cost += type_count * params["cost"]
        
        total_fault_cost += zone_fault_cost
        
        # 检查成本
        zone_check_count = check_count.get(zone_name, 0)
        zone_check_cost = per_check_cost * zone_check_count
        total_check_cost += zone_check_cost

    # 总故障成本 = 故障维护成本 + 数据丢失成本
    all_failure_cost = total_fault_cost + total_data_loss_cost

    # 总成本 = 基础成本 + 模块成本 + 总故障成本 + 检查维护成本
    total_cost = base_cost + total_module_cost + all_failure_cost + total_check_cost
    
    return total_cost, base_cost, total_module_cost, total_check_cost,total_fault_cost,total_data_loss_cost
