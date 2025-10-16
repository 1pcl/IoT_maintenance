# -*- coding: utf-8 -*-
from simulator import WSNSimulation
from utilities import MODULES, DEFAULT_PARAM_VALUES
import time
from calculate_cost import calculate_total_cost_with_simulation
import random
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import itertools
from deap import base, creator
import functools
from scene_generator import SceneGenerator
import json
import os

# 使用非交互式后端避免GUI问题
matplotlib.use('Agg')

# 定义区域列表
ZONES = ["zone_1", "zone_2", "zone_3", "zone_4"]

print(f"Total zones available: {len(ZONES)}")
scene_data = SceneGenerator.create_scene(visualize=False)
config = scene_data["config"]
zone_configs=scene_data["zone_configs"]

# 定义适应度函数（单目标：最小化总成本）
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 单目标最小化

# 定义个体（二进制列表表示模块是否被选择 + 仿真参数）
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate_individual(individual):
    """评估个体的适应度，返回完整结果字典"""
    all_selected_modules = {}
    all_simulation_params = {}
    
    try:
        # 为每个区域提取模块选择和参数
        for zone in ZONES:
            zone_modules = list(MODULES[zone].keys())
            n_modules = len(zone_modules)
            zone_start_idx = ZONES.index(zone) * (n_modules + 3)  # 每个区域有n_modules+3个参数
            
            # 提取模块选择部分
            selected_modules = [zone_modules[i] for i in range(n_modules) if individual[zone_start_idx + i] == 1]
            all_selected_modules[zone] = selected_modules
            
            # 提取仿真参数部分
            param_start_idx = zone_start_idx + n_modules
            simulation_params = {
                "preventive_check_days": int(round(individual[param_start_idx])) if individual[param_start_idx] > 0 else 1,
                "frequency_heartbeat": max(60, int(round(individual[param_start_idx + 1]))) if "heartbeat" in selected_modules else None,
                "heartbeat_loss_threshold": int(round(individual[param_start_idx + 2])) if "heartbeat" in selected_modules else None
            }
            all_simulation_params[zone] = simulation_params
        
        # 创建仿真实例
        sim = WSNSimulation(scene_data, all_selected_modules, all_simulation_params)
        
        # 计算总成本
        total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost = calculate_total_cost_with_simulation(
            sim, all_selected_modules, scene_data
        )
        
        # 计算预算超出部分
        budget = config["budget"]
        total_budget_cost = total_cost
        budget_exceed = max(0, total_budget_cost - budget)
        
        # 返回完整结果字典
        result = {
            "fitness": (float('inf'),) if budget_exceed > 0 else (total_cost,),
            "total_cost": total_cost,
            "base_cost": base_cost,
            "module_cost": module_cost,
            "check_cost": check_cost,  # 包含检查代价
            "fault_cost": fault_cost,
            "data_loss_cost": data_loss_cost,
            "feasible": budget_exceed <= 0,
            "individual": individual,  # 保留原始个体
            "selected_modules": all_selected_modules,
            "simulation_params": all_simulation_params
        }
        
        return result
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return {
            "fitness": (float('inf'),),
            "total_cost": float('inf'),
            "base_cost": 0,
            "module_cost": 0,
            "check_cost": 0,
            "fault_cost": 0,
            "data_loss_cost": 0,
            "feasible": False,
            "individual": individual,
            "selected_modules": {zone: [] for zone in ZONES},
            "simulation_params": {zone: {} for zone in ZONES}
        }

def solution_sort_key(sol_result):
    """解决方案排序函数，使用完整结果字典"""
    # 处理None值（当找不到匹配的解决方案时）
    if sol_result is None:
        return (float('inf'), float('inf'), float('inf'))  # 返回极大值确保排在最后
    
    # 计算模块代价总和
    total_module_cost = 0
    for zone in ZONES:
        if zone in sol_result["selected_modules"]:
            total_module_cost += sum(MODULES[zone][m]['cost'] for m in sol_result["selected_modules"][zone])

    # 参数偏离度计算
    total_param_deviation = 0
    for zone in ZONES:
        if zone in sol_result["simulation_params"]:
            params = sol_result["simulation_params"][zone]
            param_keys = ["preventive_check_days", "frequency_heartbeat", "heartbeat_loss_threshold"]
            param_values = [
                params["preventive_check_days"],
                params["frequency_heartbeat"] if params["frequency_heartbeat"] is not None else DEFAULT_PARAM_VALUES[zone]["frequency_heartbeat"],
                params["heartbeat_loss_threshold"] if params["heartbeat_loss_threshold"] is not None else DEFAULT_PARAM_VALUES[zone]["heartbeat_loss_threshold"]
            ]
            param_deviation = sum(abs(p - DEFAULT_PARAM_VALUES[zone][k]) for p, k in zip(param_values, param_keys))
            total_param_deviation += param_deviation

    return (sol_result["total_cost"], total_module_cost, total_param_deviation)

def solution_info_sort_key(sol_info):
    """解决方案信息排序函数，直接使用solution_info字典"""
    # 计算模块代价总和
    total_module_cost = 0
    for zone in ZONES:
        if zone in sol_info['modules']:
            total_module_cost += sum(MODULES[zone][m]['cost'] for m in sol_info['modules'][zone])
    
    # 参数偏离度计算
    total_param_deviation = 0
    for zone in ZONES:
        if zone in sol_info:
            param_keys = ["preventive_check_days", "frequency_heartbeat", "heartbeat_loss_threshold"]
            param_values = [
                sol_info[zone]['preventive_check_days'],
                sol_info[zone]['frequency_heartbeat'] if sol_info[zone]['frequency_heartbeat'] is not None else DEFAULT_PARAM_VALUES[zone]["frequency_heartbeat"],
                sol_info[zone]['heartbeat_loss_threshold'] if sol_info[zone]['heartbeat_loss_threshold'] is not None else DEFAULT_PARAM_VALUES[zone]["heartbeat_loss_threshold"]
            ]
            param_deviation = sum(abs(p - DEFAULT_PARAM_VALUES[zone][k]) for p, k in zip(param_values, param_keys))
            total_param_deviation += param_deviation

    return (sol_info['total_cost'], total_module_cost, total_param_deviation)

def neighbor_function(current_solution):
    """生成邻域解 - 多区域版本"""
    neighbor = creator.Individual(current_solution[:])  # 复制当前解
    
    # 定义参数范围（为每个区域定义）
    param_ranges = {}
    for zone in ZONES:
        check_day_min = round(zone_configs[zone]["frequency_sampling"]/(60*60*24))
        if check_day_min <= 0:
            check_day_min = 1        
        preventive_check_days_range = (check_day_min, 180)
        
        frequency_heartbeat_max = zone_configs[zone]["frequency_sampling"]
        frequency_heartbeat_min = max(1, frequency_heartbeat_max/60)
        frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
        
        heartbeat_loss_threshold_range = (3, 15)
        
        param_ranges[zone] = {
            "preventive_check_days": preventive_check_days_range,
            "frequency_heartbeat": frequency_heartbeat_range,
            "heartbeat_loss_threshold": heartbeat_loss_threshold_range
        }
    
    # 随机选择一个区域进行扰动
    selected_zone = random.choice(ZONES)
    zone_modules = list(MODULES[selected_zone].keys())
    n_modules = len(zone_modules)
    zone_start_idx = ZONES.index(selected_zone) * (n_modules + 3)
    
    # 随机选择扰动类型
    perturbation_type = random.choice([
        "flip_module", 
        "adjust_preventive_days",
        "adjust_heartbeat_freq",
        "adjust_heartbeat_threshold"
    ])
    
    # 模块部分：翻转一个随机模块
    if perturbation_type == "flip_module":
        idx = random.randint(zone_start_idx, zone_start_idx + n_modules - 1)
        neighbor[idx] = 1 - neighbor[idx]
    
    # 参数部分：调整preventive_check_days
    elif perturbation_type == "adjust_preventive_days":
        param_idx = zone_start_idx + n_modules
        perturbation = random.randint(-5, 5)
        neighbor[param_idx] = max(param_ranges[selected_zone]["preventive_check_days"][0], 
                                 min(param_ranges[selected_zone]["preventive_check_days"][1], 
                                 neighbor[param_idx] + perturbation))
    
    # 参数部分：调整frequency_heartbeat
    elif perturbation_type == "adjust_heartbeat_freq":
        param_idx = zone_start_idx + n_modules + 1
        perturbation = random.uniform(0.8, 1.2)  # 乘法扰动
        new_val = neighbor[param_idx] * perturbation
        neighbor[param_idx] = max(param_ranges[selected_zone]["frequency_heartbeat"][0], 
                                 min(param_ranges[selected_zone]["frequency_heartbeat"][1], 
                                 new_val))
    
    # 参数部分：调整heartbeat_loss_threshold
    elif perturbation_type == "adjust_heartbeat_threshold":
        param_idx = zone_start_idx + n_modules + 2
        perturbation = random.randint(-2, 2)
        neighbor[param_idx] = max(param_ranges[selected_zone]["heartbeat_loss_threshold"][0], 
                                 min(param_ranges[selected_zone]["heartbeat_loss_threshold"][1], 
                                 neighbor[param_idx] + perturbation))
    
    return neighbor

def simulated_annealing_optimization():
    """执行模拟退火优化（预算约束下的单目标优化）"""
    # 定义参数范围（为每个区域定义）
    param_ranges = {}
    for zone in ZONES:
        check_day_min = round(zone_configs[zone]["frequency_sampling"]/(60*60*24))
        if check_day_min <= 0:
            check_day_min = 1        
        preventive_check_days_range = (check_day_min, 180)
        
        frequency_heartbeat_max = zone_configs[zone]["frequency_sampling"]
        frequency_heartbeat_min = max(1, frequency_heartbeat_max/60)
        frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
        
        heartbeat_loss_threshold_range = (3, 15)
        
        param_ranges[zone] = {
            "preventive_check_days": preventive_check_days_range,
            "frequency_heartbeat": frequency_heartbeat_range,
            "heartbeat_loss_threshold": heartbeat_loss_threshold_range
        }
    
    # 个体生成函数
    def create_individual():
        individual_parts = []
        
        for zone in ZONES:
            zone_modules = list(MODULES[zone].keys())
            n_modules = len(zone_modules)
            
            # 模块选择部分（二进制）
            modules_part = [random.randint(0, 1) for _ in range(n_modules)]
            
            # 仿真参数部分
            params_part = [
                random.uniform(*param_ranges[zone]["preventive_check_days"]),
                random.uniform(*param_ranges[zone]["frequency_heartbeat"]),
                random.uniform(*param_ranges[zone]["heartbeat_loss_threshold"])
            ]
            
            individual_parts.extend(modules_part + params_part)
        
        return creator.Individual(individual_parts)
    
    # 模拟退火参数
    initial_temp = 1000.0
    final_temp = 0.1
    cooling_rate = 0.95
    max_iter = 100  # 最大迭代次数
    num_neighbors = 100  # 邻域解数量
    
    # 创建初始解
    current_solution = create_individual()
    
    # ✅ 评估初始解（单次评估）
    current_result = evaluate_individual(current_solution)
    current_cost = current_result["fitness"][0]
    
    # 记录最佳解
    best_solution = creator.Individual(current_solution[:])
    best_cost = current_cost
    best_result = current_result
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [best_cost],
        'current_cost': [current_cost],
        'temperature': [initial_temp],
        'feasible_count': [1 if current_cost != float('inf') else 0]
    }
    
    # 记录历史可行解和已见解
    feasible_solutions = []
    seen_solutions = set()
    
    if current_result["feasible"]:
        # 将个体转换为元组以便哈希
        ind_tuple = tuple(current_solution)
        seen_solutions.add(ind_tuple)
        feasible_solutions.append({
            "result": current_result,
            "cost": current_cost
        })
    
    print("Starting simulated annealing optimization...")
    print(f"Using {num_neighbors} neighbors per iteration")
    print(f"Max iterations: {max_iter}")
    start_time = time.time()
    
    # 初始温度
    temperature = initial_temp
    iteration = 0
    
    # 设置多进程评估
    num_processes = 100
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 主循环
    while temperature > final_temp and iteration < max_iter:
        # 生成多个邻域解
        neighbors = []
        for _ in range(num_neighbors):
            neighbor = neighbor_function(current_solution)
            neighbors.append(neighbor)
        
        # 并行评估所有邻域解
        results = pool.map(evaluate_individual, neighbors)
        
        # 处理结果并更新可行解列表
        for res in results:
            if res["feasible"]:
                # 检查是否已见过该解
                ind_tuple = tuple(res["individual"])
                if ind_tuple not in seen_solutions:
                    seen_solutions.add(ind_tuple)
                    feasible_solutions.append({
                        "result": res,
                        "cost": res["fitness"][0]
                    })
        
        # 从邻域解中找出最佳候选解
        candidate_result = min(results, key=lambda res: res["fitness"][0])
        candidate_solution = creator.Individual(candidate_result["individual"])
        candidate_cost = candidate_result["fitness"][0]
        
        # 计算成本差
        cost_difference = candidate_cost - current_cost
        
        # Metropolis准则：决定是否接受新解
        accept = False
        if cost_difference < 0:
            # 新解更好，总是接受
            accept = True
        elif temperature > 0:
            # 以一定概率接受较差的解（避免局部最优）
            acceptance_prob = math.exp(-cost_difference / temperature)
            if random.random() < acceptance_prob:
                accept = True
        
        # 如果接受新解，更新当前状态
        if accept:
            current_solution = creator.Individual(candidate_solution[:])
            current_cost = candidate_cost
            current_result = candidate_result
            
            # 更新最佳解
            if current_cost < best_cost:
                best_solution = creator.Individual(candidate_solution[:])
                best_cost = current_cost
                best_result = candidate_result
                print(f"Iter {iteration}: New best cost = {best_cost:.2f}, Temp = {temperature:.2f}")
        
        # 记录收敛数据
        convergence_data['min_total_cost'].append(best_cost)
        convergence_data['current_cost'].append(current_cost)
        convergence_data['temperature'].append(temperature)
        convergence_data['feasible_count'].append(len(feasible_solutions))
        
        # 降温
        temperature *= cooling_rate
        iteration += 1
        
        # 打印进度
        print(f"Iter {iteration}: Best cost = {best_cost:.2f}, Current cost = {current_cost:.2f}, "
              f"Temp = {temperature:.2f}, Feasible = {len(feasible_solutions)}")
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 从可行解中选择最佳解
    if feasible_solutions:
        # 使用多标准排序
        feasible_solutions.sort(key=lambda x: solution_sort_key(x["result"]))
        best_feasible = feasible_solutions[0]["result"]
    else:
        # 如果没有可行解，使用当前最佳解
        best_feasible = best_result
    
    # 填充收敛数据结构以保持兼容
    convergence_data_full = {
        'min_total_cost': convergence_data['min_total_cost'],
        'current_cost': convergence_data['current_cost'],
        'avg_total_cost': convergence_data['min_total_cost'],  # 模拟退火没有平均值，用最小值替代
        'feasible_count': convergence_data['feasible_count'],
        'mutation_rate': [cooling_rate] * len(convergence_data['min_total_cost']),
        'diversity': [0] * len(convergence_data['min_total_cost']),  # 模拟退火没有多样性概念
        'genetic_diversity': [0] * len(convergence_data['min_total_cost']),  # 模拟退火没有基因多样性
        'temperature': convergence_data['temperature']  # 添加温度数据
    }
    
    print(f"SA completed in {iteration} iterations. Total evaluations: {iteration * num_neighbors}")
    print(f"Best cost: {best_feasible['total_cost']:.2f}, Time: {elapsed:.2f}s")
    return best_feasible, feasible_solutions, elapsed, num_processes, convergence_data_full

def plot_convergence(convergence_data, elapsed_time, num_procs):
    """分别绘制模拟退火的各类收敛图"""
    # 确保目录存在
    os.makedirs("results2", exist_ok=True)
    
    iterations = range(len(convergence_data['min_total_cost']))
    
    # 1. 总成本收敛图
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convergence_data['min_total_cost'], 'b-', label='Min Total Cost')
    
    # 确保current_cost存在才绘制
    if 'current_cost' in convergence_data:
        plt.plot(iterations, convergence_data['current_cost'], 'g--', label='Current Cost')
    else:
        print("Warning: 'current_cost' not found in convergence_data")
    
    plt.xlabel('Iteration')
    plt.ylabel('Total Cost')
    plt.title(f'Total Cost Convergence\n(Time: {elapsed_time:.2f}s, Processes: {num_procs})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results2/sa_convergence_total_cost.png')
    print("Saved: results2/sa_convergence_total_cost.png")
    plt.close()

    # 2. 温度变化曲线
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convergence_data['temperature'], 'm-', label='Temperature')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.title('Temperature Schedule')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results2/sa_convergence_temperature.png')
    print("Saved: results2/sa_convergence_temperature.png")
    plt.close()

    # 3. 可行解数量变化
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convergence_data['feasible_count'], 'g-', label='Feasible Solutions')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Solutions')
    plt.title('Cumulative Feasible Solutions')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results2/sa_convergence_feasible_count.png')
    print("Saved: results2/sa_convergence_feasible_count.png")
    plt.close()

    # 4. 冷却率变化（恒定）
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, convergence_data['mutation_rate'], 'c-', label='Cooling Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Cooling Rate')
    plt.title('Cooling Rate (Constant)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results2/sa_convergence_cooling_rate.png')
    print("Saved: results2/sa_convergence_cooling_rate.png")
    plt.close()

def extract_solution_info(solution_data):
    """从解决方案数据中提取信息"""
    return {
        "modules": solution_data["selected_modules"],
        "zone_1": solution_data["simulation_params"]["zone_1"],
        "zone_2": solution_data["simulation_params"]["zone_2"],
        "zone_3": solution_data["simulation_params"]["zone_3"],
        "zone_4": solution_data["simulation_params"]["zone_4"],
        "total_cost": solution_data["total_cost"],
        "base_cost": solution_data["base_cost"],
        "module_cost": solution_data["module_cost"],
        "check_cost": solution_data["check_cost"],  # 包含检查代价
        "fault_cost": solution_data["fault_cost"],
        "data_loss_cost": solution_data["data_loss_cost"],
        "budget": config["budget"],
        "encoded_solution": solution_data["individual"]  # 原始个体
    }

def save_convergence_data(convergence_data, elapsed_time, num_procs):
    """保存收敛数据到JSON文件"""
    # 确保目录存在
    os.makedirs("results2", exist_ok=True)
    
    # 准备要保存的数据
    data_to_save = {
        "min_total_cost": convergence_data["min_total_cost"],
        "current_cost": convergence_data["current_cost"],
        "avg_total_cost": convergence_data["avg_total_cost"],
        "feasible_count": convergence_data["feasible_count"],
        "mutation_rate": convergence_data["mutation_rate"],
        "diversity": convergence_data["diversity"],
        "genetic_diversity": convergence_data["genetic_diversity"],
        "temperature": convergence_data["temperature"],
        "elapsed_time": elapsed_time,
        "num_procs": num_procs
    }
    
    # 保存到文件
    with open("results2/sa_convergence_data.json", "w") as f:
        json.dump(data_to_save, f, indent=4)
    
    print("Saved: results2/sa_convergence_data.json")

def replot_from_saved_data():
    """从保存的数据重新绘制图表"""
    try:
        with open("results2/sa_convergence_data.json", "r") as f:
            plot_data = json.load(f)
        
        convergence_data = {
            "min_total_cost": plot_data["min_total_cost"],
            "current_cost": plot_data["current_cost"],
            "avg_total_cost": plot_data["avg_total_cost"],
            "feasible_count": plot_data["feasible_count"],
            "mutation_rate": plot_data["mutation_rate"],
            "diversity": plot_data["diversity"],
            "genetic_diversity": plot_data["genetic_diversity"],
            "temperature": plot_data["temperature"]
        }
        
        plot_convergence(convergence_data, plot_data["elapsed_time"], plot_data["num_procs"])
        print("Successfully replotted from saved data")
    except FileNotFoundError:
        print("No saved data found. Please run the optimization first.")
    except Exception as e:
        print(f"Error replotting: {e}")

if __name__ == "__main__":
    # 检查是否只需要重新绘图
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        replot_from_saved_data()
        sys.exit(0)
    
    # 运行模拟退火优化
    print("Starting simulated annealing optimization...")
    
    try:
        # 运行优化
        best_feasible, feasible_solutions, elapsed_time, num_procs, convergence_data = simulated_annealing_optimization()
        
        # 保存收敛数据
        save_convergence_data(convergence_data, elapsed_time, num_procs)
        
        # 绘制收敛图
        plot_convergence(convergence_data, elapsed_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_feasible)

        # 创建结果目录
        os.makedirs("results2", exist_ok=True)
        
        # 写入结果文件
        with open("results2/sa_optimization_results.txt", "w") as f:
            f.write("Simulated Annealing Optimization Results (Budget-Constrained)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Optimization Time: {elapsed_time:.2f} seconds | Processes Used: {num_procs}\n")
            f.write(f"Total Evaluations: {len(convergence_data['min_total_cost']) * 100}\n")
            f.write(f"Budget: {best_solution['budget']}\n")
            f.write(f"Total Cost: {best_solution['total_cost']:.2f}\n\n")
            
            # 最佳解决方案
            f.write("Best Solution:\n")
            f.write("=" * 80 + "\n")
            
            # 为每个区域输出模块和参数
            for zone in ZONES:
                f.write(f"\n{zone.upper()}:\n")
                f.write(f"  Modules: {best_solution['modules'][zone]}\n")
                f.write(f"  Preventive Check Days: {best_solution[zone]['preventive_check_days']} days\n")
                if best_solution[zone]['frequency_heartbeat'] is not None:
                    f.write(f"  Heartbeat Frequency: {best_solution[zone]['frequency_heartbeat']} seconds\n")
                    f.write(f"  Heartbeat Loss Threshold: {best_solution[zone]['heartbeat_loss_threshold']}\n")
            
            f.write("\nCost Breakdown:\n")
            f.write(f"  Base Cost:      {best_solution['base_cost']:.2f}\n")
            f.write(f"  Module Cost:    {best_solution['module_cost']:.2f}\n")
            f.write(f"  Check Cost:     {best_solution['check_cost']:.2f}\n")  # 添加检查代价
            f.write(f"  Fault Cost:     {best_solution['fault_cost']:.2f}\n")
            f.write(f"  Data Loss Cost: {best_solution['data_loss_cost']:.2f}\n")
            f.write(f"  Total Cost:     {best_solution['total_cost']:.2f}\n")
            f.write("-" * 80 + "\n\n")
            
            # 其他可行解
            if feasible_solutions:
                # 提取解决方案信息用于排序
                solutions_info = []
                for sol in feasible_solutions:
                    sol_info = extract_solution_info(sol["result"])
                    solutions_info.append(sol_info)
                
                # 修复排序问题：使用新的排序函数
                solutions_info.sort(key=solution_info_sort_key)
                
                f.write(f"Other Feasible Solutions ({len(feasible_solutions)} total):\n")
                f.write("=" * 80 + "\n")
                for i, sol in enumerate(solutions_info[:10]):
                    f.write(f"Solution {i+1} (Cost: {sol['total_cost']:.2f}):\n")
                    for zone in ZONES:
                        f.write(f"  {zone.upper()}:\n")
                        f.write(f"    Modules: {sol['modules'][zone]}\n")
                        f.write(f"    Preventive Check Days: {sol[zone]['preventive_check_days']} days\n")
                        if sol[zone]['frequency_heartbeat'] is not None:
                            f.write(f"    Heartbeat Frequency: {sol[zone]['frequency_heartbeat']} seconds\n")
                            f.write(f"    Heartbeat Loss Threshold: {sol[zone]['heartbeat_loss_threshold']}\n")
                    f.write("  Cost Breakdown:\n")
                    f.write(f"    Base Cost:      {sol['base_cost']:.2f}\n")
                    f.write(f"    Module Cost:    {sol['module_cost']:.2f}\n")
                    f.write(f"    Check Cost:     {sol['check_cost']:.2f}\n")  # 添加检查代价
                    f.write(f"    Fault Cost:     {sol['fault_cost']:.2f}\n")
                    f.write(f"    Data Loss Cost: {sol['data_loss_cost']:.2f}\n")
                    f.write(f"    Total Cost:     {sol['total_cost']:.2f}\n")
                    f.write("-" * 80 + "\n")
            else:
                f.write("No other feasible solutions found.\n")
        
        # 打印最佳解决方案
        print("\nBest Solution Found:")
        for zone in ZONES:
            print(f"\n{zone.upper()}:")
            print(f"  Modules: {best_solution['modules'][zone]}")
            print(f"  Preventive Check Days: {best_solution[zone]['preventive_check_days']} days")
            if best_solution[zone]['frequency_heartbeat'] is not None:
                print(f"  Heartbeat Frequency: {best_solution[zone]['frequency_heartbeat']} seconds")
                print(f"  Heartbeat Loss Threshold: {best_solution[zone]['heartbeat_loss_threshold']}")
        print("\nCost Breakdown:")
        print(f"  Base Cost:      {best_solution['base_cost']:.2f}")
        print(f"  Module Cost:    {best_solution['module_cost']:.2f}")
        print(f"  Check Cost:     {best_solution['check_cost']:.2f}")  # 添加检查代价
        print(f"  Fault Cost:     {best_solution['fault_cost']:.2f}")
        print(f"  Data Loss Cost: {best_solution['data_loss_cost']:.2f}")
        print(f"  Total Cost:     {best_solution['total_cost']:.2f}")
        print(f"  Budget: {best_solution['budget']}")
                
        # 打印执行时间
        print(f"\nSA optimization completed! Time: {elapsed_time:.2f} seconds | Processes: {num_procs}")
        print(f"Total evaluations: {len(convergence_data['min_total_cost']) * 100}")
        print(f"Feasible solutions found: {len(feasible_solutions) if feasible_solutions else 0}")
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)