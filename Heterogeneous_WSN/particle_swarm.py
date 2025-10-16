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
import sys
import itertools
import math
import functools
import json
import os
from deap import base, creator
from scene_generator import SceneGenerator

# 使用非交互式后端避免GUI问题
matplotlib.use('Agg')

# 定义区域列表
ZONES = ["zone_1", "zone_2", "zone_3", "zone_4"]

print(f"Total zones available: {len(ZONES)}")
scene_data = SceneGenerator.create_scene(visualize=False)
config=scene_data["config"]
zone_configs=scene_data["zone_configs"]

def compare_individuals(ind1, ind2):
    """多标准排序：
    1. 总代价最小
    2. 模块代价总和最小
    3. 参数偏离常规值最小
    """
    cost1, cost2 = ind1.fitness.values[0], ind2.fitness.values[0]
    if cost1 < cost2:
        return -1
    elif cost1 > cost2:
        return 1

    # 模块代价总和
    total_module_cost1 = 0
    total_module_cost2 = 0
    for zone in ZONES:
        zone_modules = list(MODULES[zone].keys())
        n_modules = len(zone_modules)
        zone_start_idx = ZONES.index(zone) * (n_modules + 3)  # 每个区域有n_modules+3个参数
        
        selected_modules1 = [zone_modules[i] for i in range(n_modules) if ind1[zone_start_idx + i] >= 0.5]
        selected_modules2 = [zone_modules[i] for i in range(n_modules) if ind2[zone_start_idx + i] >= 0.5]
        
        total_module_cost1 += sum(MODULES[zone][m]['cost'] for m in selected_modules1)
        total_module_cost2 += sum(MODULES[zone][m]['cost'] for m in selected_modules2)

    if total_module_cost1 < total_module_cost2:
        return -1
    elif total_module_cost1 > total_module_cost2:
        return 1

    # 参数偏离程度
    total_deviation1 = 0
    total_deviation2 = 0
    for zone in ZONES:
        zone_modules = list(MODULES[zone].keys())
        n_modules = len(zone_modules)
        zone_start_idx = ZONES.index(zone) * (n_modules + 3)
        
        param1 = ind1[zone_start_idx + n_modules:zone_start_idx + n_modules + 3]
        param2 = ind2[zone_start_idx + n_modules:zone_start_idx + n_modules + 3]
        
        param_keys = list(DEFAULT_PARAM_VALUES[zone].keys())
        deviation1 = 0
        deviation2 = 0
        
        # 总是存在的参数
        deviation1 += abs(param1[0] - DEFAULT_PARAM_VALUES[zone]['preventive_check_days'])
        deviation2 += abs(param2[0] - DEFAULT_PARAM_VALUES[zone]['preventive_check_days'])
        
        # 只有当heartbeat模块被选中时才计算相关参数
        selected_modules1 = [zone_modules[i] for i in range(n_modules) if ind1[zone_start_idx + i] >= 0.5]
        selected_modules2 = [zone_modules[i] for i in range(n_modules) if ind2[zone_start_idx + i] >= 0.5]
        
        if "heartbeat" in selected_modules1:
            deviation1 += abs(param1[1] - DEFAULT_PARAM_VALUES[zone]['frequency_heartbeat'])
            deviation1 += abs(param1[2] - DEFAULT_PARAM_VALUES[zone]['heartbeat_loss_threshold'])
        if "heartbeat" in selected_modules2:
            deviation2 += abs(param2[1] - DEFAULT_PARAM_VALUES[zone]['frequency_heartbeat'])
            deviation2 += abs(param2[2] - DEFAULT_PARAM_VALUES[zone]['heartbeat_loss_threshold'])
        
        total_deviation1 += deviation1
        total_deviation2 += deviation2

    if total_deviation1 < total_deviation2:
        return -1
    elif total_deviation1 > total_deviation2:
        return 1

    return 0  # 完全相等

def solution_sort_key(sol):
    # 计算模块代价总和
    total_module_cost = 0
    for zone in ZONES:
        if zone in sol['modules']:
            total_module_cost += sum(MODULES[zone][m]['cost'] for m in sol['modules'][zone])

    # 参数偏离度计算
    total_param_deviation = 0
    for zone in ZONES:
        if zone in sol:
            param_keys = list(DEFAULT_PARAM_VALUES[zone].keys())
            param_values = [
                sol[zone]['preventive_check_days'],
                sol[zone]['frequency_heartbeat'] if sol[zone]['frequency_heartbeat'] is not None else DEFAULT_PARAM_VALUES[zone]["frequency_heartbeat"],
                sol[zone]['heartbeat_loss_threshold'] if sol[zone]['heartbeat_loss_threshold'] is not None else DEFAULT_PARAM_VALUES[zone]["heartbeat_loss_threshold"]
            ]
            param_deviation = 0
            param_deviation += abs(param_values[0] - DEFAULT_PARAM_VALUES[zone]['preventive_check_days'])
            if "heartbeat" in sol['modules'][zone]:
                param_deviation += abs(param_values[1] - DEFAULT_PARAM_VALUES[zone]['frequency_heartbeat'])
                param_deviation += abs(param_values[2] - DEFAULT_PARAM_VALUES[zone]['heartbeat_loss_threshold'])
            total_param_deviation += param_deviation

    return (sol['total_cost'], total_module_cost, total_param_deviation)

# 定义适应度函数（单目标：最小化总成本）
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 单目标最小化

# 定义个体（二进制列表表示模块是否被选择 + 仿真参数）
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate_individual(individual):
    """评估个体的适应度"""
    all_selected_modules = {}
    all_simulation_params = {}
    
    try:
        # 为每个区域提取模块选择和参数
        for zone in ZONES:
            zone_modules = list(MODULES[zone].keys())
            n_modules = len(zone_modules)
            zone_start_idx = ZONES.index(zone) * (n_modules + 3)  # 每个区域有n_modules+3个参数
            
            # 提取模块选择部分 - 使用0.5作为阈值将连续值转换为二进制
            selected_modules = [zone_modules[i] for i in range(n_modules) if individual[zone_start_idx + i] >= 0.5]
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
        
        # 创建成本分解字典
        cost_breakdown = {
            "total_cost": total_cost,
            "base_cost": base_cost,
            "module_cost": module_cost,
            "check_cost": check_cost,  # 新增检查成本
            "fault_cost": fault_cost,
            "data_loss_cost": data_loss_cost
        }
        
        # 保持硬约束，预算超支返回实际值但标记为不可行
        return (total_cost,), cost_breakdown, budget_exceed
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return (float('inf'),), {
            "total_cost": float('inf'),
            "base_cost": 0,
            "module_cost": 0,
            "check_cost": 0,
            "fault_cost": 0,
            "data_loss_cost": 0
        }, float('inf')  # 标记为严重不可行

def calculate_genetic_diversity(population_positions):
    """计算种群的基因多样性（基于汉明距离）"""
    if len(population_positions) < 2:
        return 0.0
    
    diversity = 0.0
    count = 0
    
    # 计算所有个体两两之间的汉明距离（仅模块部分）
    for i, j in itertools.combinations(range(len(population_positions)), 2):
        # 为每个区域计算模块部分的汉明距离
        hamming_dist = 0
        for zone in ZONES:
            zone_modules = list(MODULES[zone].keys())
            n_modules = len(zone_modules)
            zone_start_idx = ZONES.index(zone) * (n_modules + 3)
            
            # 将连续值转换为二进制
            ind1 = [1 if x >= 0.5 else 0 for x in population_positions[i][zone_start_idx:zone_start_idx + n_modules]]
            ind2 = [1 if x >= 0.5 else 0 for x in population_positions[j][zone_start_idx:zone_start_idx + n_modules]]
            
            # 计算汉明距离（不同基因的数量）
            hamming_dist += sum(g1 != g2 for g1, g2 in zip(ind1, ind2))
        
        diversity += hamming_dist
        count += 1
    
    # 返回平均汉明距离
    return diversity / count if count > 0 else 0.0

class Particle:
    """粒子群优化中的粒子类"""
    def __init__(self, position):
        self.position = position
        self.velocity = [0.0] * len(position)
        self.best_position = position[:]
        self.best_fitness = float('inf')
        self.fitness = float('inf')
        self.cost_breakdown = None  # 存储成本分解信息
        self.budget_exceed = float('inf')  # 存储预算超支值
        self.best_cost_breakdown = None  # 历史最优的成本分解
        self.best_budget_exceed = float('inf')  # 历史最优的预算超支值
    
    def update_position(self, bounds):
        """更新粒子位置，考虑边界约束"""
        for i in range(len(self.position)):
            # 更新位置
            self.position[i] += self.velocity[i]
            
            # 应用边界约束
            if i < len(bounds):
                min_bound, max_bound = bounds[i]
                self.position[i] = max(min_bound, min(max_bound, self.position[i]))
    
    def update_best(self):
        """更新粒子的最佳位置"""
        # 优先选择预算不超支的解
        if self.budget_exceed == 0 and self.best_budget_exceed > 0:
            self.best_position = self.position[:]
            self.best_fitness = self.fitness
            self.best_budget_exceed = self.budget_exceed
            self.best_cost_breakdown = self.cost_breakdown  # 保存历史最优的成本分解
        elif self.budget_exceed == 0 and self.best_budget_exceed == 0:
            # 两者都不超支，选择总成本更小的
            if self.fitness < self.best_fitness:
                self.best_position = self.position[:]
                self.best_fitness = self.fitness
                self.best_budget_exceed = self.budget_exceed
                self.best_cost_breakdown = self.cost_breakdown  # 保存历史最优的成本分解
        elif self.budget_exceed > 0 and self.best_budget_exceed > 0:
            # 两者都超支，选择超支更小的
            if self.budget_exceed < self.best_budget_exceed:
                self.best_position = self.position[:]
                self.best_fitness = self.fitness
                self.best_budget_exceed = self.budget_exceed
                self.best_cost_breakdown = self.cost_breakdown  # 保存历史最优的成本分解
        else:
            # 当前解超支但历史解不超支，不更新
            pass

def particle_swarm_optimization(num_processes=100):
    """执行粒子群优化（预算约束下的单目标优化）"""
    # 定义参数范围（为每个区域定义）
    param_ranges = {}
    for zone in ZONES:
        check_day_min = round(zone_configs[zone]["frequency_sampling"]/(60*60*24))
        if check_day_min <= 0:
            check_day_min = 1        
        preventive_check_days_range = (check_day_min, 180)
        
        frequency_heartbeat_max = zone_configs[zone]["frequency_sampling"]
        frequency_heartbeat_min = max(60, frequency_heartbeat_max / 60)  # 确保最小值合理
        frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
        
        heartbeat_loss_threshold_range = (3, 15)
        
        param_ranges[zone] = {
            "preventive_check_days": preventive_check_days_range,
            "frequency_heartbeat": frequency_heartbeat_range,
            "heartbeat_loss_threshold": heartbeat_loss_threshold_range
        }
    
    # 定义搜索空间边界
    bounds = []
    
    # 为每个区域添加模块选择和参数边界
    for zone in ZONES:
        zone_modules = list(MODULES[zone].keys())
        n_modules = len(zone_modules)
        
        # 模块选择部分（二进制，但PSO中我们使用连续值并通过sigmoid转换）
        for _ in range(n_modules):
            bounds.append((0, 1))  # 二进制变量边界
        
        # 仿真参数部分
        bounds.append(param_ranges[zone]["preventive_check_days"])
        bounds.append(param_ranges[zone]["frequency_heartbeat"])
        bounds.append(param_ranges[zone]["heartbeat_loss_threshold"])
    
    # 粒子群参数 - 使用固定值
    pop_size = 100
    ngen = 100
    w = 0.729  # 固定惯性权重
    c1 = 1.49445
    c2 = 1.49445
    v_max = 0.2  # 固定最大速度
    
    # 创建初始种群
    def create_particle():
        particle_parts = []
        
        for zone in ZONES:
            zone_modules = list(MODULES[zone].keys())
            n_modules = len(zone_modules)
            
            # 模块选择部分（二进制）
            modules_part = [random.uniform(0, 1) for _ in range(n_modules)]
            
            # 仿真参数部分
            params_part = [
                random.uniform(*param_ranges[zone]["preventive_check_days"]),
                random.uniform(*param_ranges[zone]["frequency_heartbeat"]),
                random.uniform(*param_ranges[zone]["heartbeat_loss_threshold"])
            ]
            
            particle_parts.extend(modules_part + params_part)
        
        particle = Particle(particle_parts)
        particle.best_budget_exceed = float('inf')  # 初始化历史最佳预算超支值
        return particle
    
    population = [create_particle() for _ in range(pop_size)]
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 评估初始种群
    print(f"Evaluating initial population (using {num_processes} processes)...")
    positions = [p.position for p in population]
    results = pool.map(evaluate_individual, positions)
    
    # 更新粒子适应度、成本分解和预算超支值
    for i, p in enumerate(population):
        fitness, cost_breakdown, budget_exceed = results[i]
        p.fitness = fitness[0]
        p.cost_breakdown = cost_breakdown
        p.budget_exceed = budget_exceed
        p.update_best()
    
    # 全局最佳粒子
    global_best_particle = min(
        population,
        key=lambda p: (p.best_budget_exceed, p.best_fitness)  # 优先选不超支的解
    )
    global_best_position = global_best_particle.best_position[:]
    global_best_fitness = global_best_particle.best_fitness
    global_best_budget_exceed = global_best_particle.best_budget_exceed
    
    # 创建全局最优解跟踪器 - 在整个优化过程中保持最佳解
    global_best_overall = {
        'position': global_best_position[:],
        'fitness': global_best_fitness,
        'budget_exceed': global_best_budget_exceed,
        'cost_breakdown': global_best_particle.best_cost_breakdown
    }
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [],
        'avg_total_cost': [],
        'feasible_count': [],
        'diversity': [],
        'genetic_diversity': []
    }
    
    # 记录初始代
    total_costs = [p.fitness for p in population]
    feasible_count = sum(1 for p in population if p.budget_exceed == 0)
    min_cost = min(total_costs)
    
    convergence_data['min_total_cost'].append(min_cost)
    convergence_data['avg_total_cost'].append(
        sum(c for c in total_costs) / pop_size)
    convergence_data['feasible_count'].append(feasible_count)
    
    # 计算初始多样性和基因多样性
    feasible_costs = [p.fitness for p in population if p.budget_exceed == 0]
    diversity = np.std(feasible_costs) if feasible_costs else 0
    genetic_diversity = calculate_genetic_diversity(positions)
    
    convergence_data['diversity'].append(diversity)
    convergence_data['genetic_diversity'].append(genetic_diversity)
    
    # 运行PSO优化
    print("Starting PSO optimization...")
    start_time = time.time()
    
    for gen in range(ngen):
        # 更新每个粒子
        for p in population:
            # 更新速度
            for i in range(len(p.velocity)):
                r1 = random.random()
                r2 = random.random()
                
                # 速度更新公式
                cognitive = c1 * r1 * (p.best_position[i] - p.position[i])
                social = c2 * r2 * (global_best_position[i] - p.position[i])
                p.velocity[i] = w * p.velocity[i] + cognitive + social
                
                # 限制速度范围
                min_bound, max_bound = bounds[i]
                range_size = max_bound - min_bound
                p.velocity[i] = max(-v_max * range_size, min(v_max * range_size, p.velocity[i]))
            
            # 更新位置
            p.update_position(bounds)
        
        # 评估所有粒子
        positions = [p.position for p in population]
        results = pool.map(evaluate_individual, positions)
        
        # 更新粒子适应度、成本分解和预算超支值
        for i, p in enumerate(population):
            fitness, cost_breakdown, budget_exceed = results[i]
            p.fitness = fitness[0]
            p.cost_breakdown = cost_breakdown
            p.budget_exceed = budget_exceed
            p.update_best()
            
            # 检查并更新全局最优解
            if p.best_budget_exceed < global_best_overall['budget_exceed'] or \
               (p.best_budget_exceed == global_best_overall['budget_exceed'] and \
                p.best_fitness < global_best_overall['fitness']):
                
                # 更新全局最优解
                global_best_overall = {
                    'position': p.best_position[:],
                    'fitness': p.best_fitness,
                    'budget_exceed': p.best_budget_exceed,
                    'cost_breakdown': p.best_cost_breakdown
                }
        
        # 记录当前代数据
        total_costs = [p.fitness for p in population]
        feasible_count = sum(1 for p in population if p.budget_exceed == 0)
        current_min = min(total_costs)
        
        convergence_data['min_total_cost'].append(current_min)
        convergence_data['avg_total_cost'].append(
            sum(c for c in total_costs) / pop_size)
        convergence_data['feasible_count'].append(feasible_count)
        
        # 计算并记录种群多样性和基因多样性
        feasible_costs = [p.fitness for p in population if p.budget_exceed == 0]
        diversity = np.std(feasible_costs) if feasible_costs else 0
        genetic_diversity = calculate_genetic_diversity(positions)
        
        convergence_data['diversity'].append(diversity)
        convergence_data['genetic_diversity'].append(genetic_diversity)
        
        # 打印进度
        print(f"Gen {gen+1}/{ngen}: Min Cost={current_min:.2f}, "
              f"Feasible={feasible_count}/{pop_size}, "
              f"Diversity={diversity:.1f}, "
              f"GeneticDiv={genetic_diversity:.1f}")
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 使用全局最优解创建最佳个体
    best_individual = creator.Individual(global_best_overall['position'])
    best_individual.fitness.values = (global_best_overall['fitness'],)
    best_individual.cost_breakdown = global_best_overall['cost_breakdown']
    
    # 创建最终种群并设置适应度值
    final_population = []
    for p in population:
        ind = creator.Individual(p.position)
        ind.fitness.values = (p.fitness,)
        ind.cost_breakdown = p.cost_breakdown
        final_population.append(ind)
    
    return best_individual, final_population, elapsed, num_processes, convergence_data

def plot_convergence(convergence_data, elapsed_time, num_procs):
    """绘制PSO收敛图并保存数据"""
    # 确保目录存在
    os.makedirs("results3", exist_ok=True)
    
    # 保存绘图数据到JSON文件
    plot_data = {
        "min_total_cost": convergence_data['min_total_cost'],
        "avg_total_cost": convergence_data['avg_total_cost'],
        "feasible_count": convergence_data['feasible_count'],
        "diversity": convergence_data['diversity'],
        "genetic_diversity": convergence_data['genetic_diversity'],
        "elapsed_time": elapsed_time,
        "num_procs": num_procs
    }
    
    with open("results3/pso_convergence_data.json", "w") as f:
        json.dump(plot_data, f, indent=4)
    
    # 1. 总成本收敛图
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(convergence_data['min_total_cost'], 'b-', label='Min Total Cost', linewidth=2)
    plt.plot(convergence_data['avg_total_cost'], 'r--', label='Avg Total Cost', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Total Cost', fontsize=16)
    plt.title(f'PSO Optimization Convergence\n(Time: {elapsed_time:.2f}s, Processes: {num_procs})', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results3/pso_convergence_total_cost.png', dpi=300)
    print("Saved: results3/pso_convergence_total_cost.png")
    plt.close()

    # 2. 可行解数量变化
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(convergence_data['feasible_count'], 'g-', label='Feasible Solutions', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Number of Solutions', fontsize=16)
    plt.title('Feasible Solutions Evolution', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results3/pso_convergence_feasible_count.png', dpi=300)
    print("Saved: results3/pso_convergence_feasible_count.png")
    plt.close()

    # 3. 种群多样性
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(convergence_data['diversity'], 'c-', label='Fitness Diversity (Std Dev)', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Diversity', fontsize=16)
    plt.title('Fitness Diversity Over Generations', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results3/pso_convergence_diversity.png', dpi=300)
    print("Saved: results3/pso_convergence_diversity.png")
    plt.close()

    # 4. 基因多样性
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(convergence_data['genetic_diversity'], 'y-', label='Genetic Diversity (Avg Hamming Dist)', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Genetic Diversity', fontsize=16)
    plt.title('Genetic Diversity Over Generations', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results3/pso_convergence_genetic_diversity.png', dpi=300)
    print("Saved: results3/pso_convergence_genetic_diversity.png")
    plt.close()

def replot_from_saved_data():
    """从保存的数据重新绘制图表"""
    try:
        with open("results3/pso_convergence_data.json", "r") as f:
            plot_data = json.load(f)
        
        convergence_data = {
            'min_total_cost': plot_data["min_total_cost"],
            'avg_total_cost': plot_data["avg_total_cost"],
            'feasible_count': plot_data["feasible_count"],
            'diversity': plot_data["diversity"],
            'genetic_diversity': plot_data["genetic_diversity"]
        }
        
        plot_convergence(convergence_data, plot_data["elapsed_time"], plot_data["num_procs"])
        print("Successfully replotted from saved data")
    except FileNotFoundError:
        print("No saved data found. Please run the optimization first.")
    except Exception as e:
        print(f"Error replotting: {e}")

def extract_solution_info(individual):
    """从个体中提取解决方案信息"""
    all_selected_modules = {}
    all_simulation_params = {}
    
    try:
        # 为每个区域提取模块选择和参数
        for zone in ZONES:
            zone_modules = list(MODULES[zone].keys())
            n_modules = len(zone_modules)
            zone_start_idx = ZONES.index(zone) * (n_modules + 3)
            
            # 提取模块选择 - 使用0.5作为阈值将连续值转换为二进制
            selected_modules = [zone_modules[i] for i in range(n_modules) if individual[zone_start_idx + i] >= 0.5]
            all_selected_modules[zone] = selected_modules
            
            # 提取仿真参数
            param_start_idx = zone_start_idx + n_modules
            preventive_check_days = int(round(individual[param_start_idx]))
            frequency_heartbeat = int(round(individual[param_start_idx + 1])) if "heartbeat" in selected_modules else None
            heartbeat_loss_threshold = int(round(individual[param_start_idx + 2])) if "heartbeat" in selected_modules else None
            
            all_simulation_params[zone] = {
                "preventive_check_days": preventive_check_days,
                "frequency_heartbeat": frequency_heartbeat,
                "heartbeat_loss_threshold": heartbeat_loss_threshold
            }
        
        # 获取适应度值和成本分解
        total_cost = individual.fitness.values[0] if hasattr(individual, 'fitness') else float('inf')
        
        # 使用存储的成本分解信息
        if hasattr(individual, 'cost_breakdown'):
            cost_info = individual.cost_breakdown
        else:
            cost_info = {
                "total_cost": total_cost,
                "base_cost": 0,
                "module_cost": 0,
                "check_cost": 0,
                "fault_cost": 0,
                "data_loss_cost": 0
            }
        
        return {
            "modules": all_selected_modules,
            "zone_1": all_simulation_params["zone_1"],
            "zone_2": all_simulation_params["zone_2"],
            "zone_3": all_simulation_params["zone_3"],
            "zone_4": all_simulation_params["zone_4"],
            "total_cost": cost_info["total_cost"],
            "base_cost": cost_info["base_cost"],
            "module_cost": cost_info["module_cost"],
            "check_cost": cost_info["check_cost"],  # 新增检查成本
            "fault_cost": cost_info["fault_cost"],
            "data_loss_cost": cost_info["data_loss_cost"],
            "budget": config["budget"]
        }
    except Exception as e:
        print(f"Error extracting solution info: {e}")
        return {
            "modules": {zone: [] for zone in ZONES},
            "zone_1": {"preventive_check_days": 1, "frequency_heartbeat": None, "heartbeat_loss_threshold": None},
            "zone_2": {"preventive_check_days": 1, "frequency_heartbeat": None, "heartbeat_loss_threshold": None},
            "zone_3": {"preventive_check_days": 1, "frequency_heartbeat": None, "heartbeat_loss_threshold": None},
            "zone_4": {"preventive_check_days": 1, "frequency_heartbeat": None, "heartbeat_loss_threshold": None},
            "total_cost": float('inf'),
            "base_cost": 0,
            "module_cost": 0,
            "check_cost": 0,
            "fault_cost": 0,
            "data_loss_cost": 0,
            "budget": config["budget"]
        }

if __name__ == "__main__":
    # 检查是否只需要重新绘图
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        replot_from_saved_data()
        sys.exit(0)
    
    # 运行粒子群优化
    print("Starting multi-process PSO optimization...")
    
    try:
        best_individual, population, elapsed_time, num_procs, convergence_data = particle_swarm_optimization(num_processes=100)
        
        # 绘制收敛图
        plot_convergence(convergence_data, elapsed_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_individual)
        
        # 创建结果目录
        os.makedirs("results3", exist_ok=True)
        
        # 写入结果文件
        with open("results3/pso_optimization_results.txt", "w") as f:
            f.write("Particle Swarm Optimization Results (Budget-Constrained)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Optimization Time: {elapsed_time:.2f} seconds | Processes Used: {num_procs}\n")
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
            f.write(f"  Check Cost:     {best_solution['check_cost']:.2f}\n")  # 新增检查成本
            f.write(f"  Fault Cost:     {best_solution['fault_cost']:.2f}\n")
            f.write(f"  Data Loss Cost: {best_solution['data_loss_cost']:.2f}\n")
            f.write(f"  Total Cost:     {best_solution['total_cost']:.2f}\n")
            f.write("-" * 80 + "\n\n")
            
            # 其他可行解（最多前10个）
            feasible_solutions = []
            for ind in population:
                solution = extract_solution_info(ind)
                # 只考虑预算不超支的解
                if solution['total_cost'] <= solution['budget']:
                    feasible_solutions.append(solution)
            
            # 按总成本排序
            feasible_solutions.sort(key=solution_sort_key)
            
            f.write(f"Other Feasible Solutions ({len(feasible_solutions)} total):\n")
            f.write("=" * 80 + "\n")
            for i, sol in enumerate(feasible_solutions[:10]):
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
                f.write(f"    Check Cost:     {sol['check_cost']:.2f}\n")  # 新增检查成本
                f.write(f"    Fault Cost:     {sol['fault_cost']:.2f}\n")
                f.write(f"    Data Loss Cost: {sol['data_loss_cost']:.2f}\n")
                f.write(f"    Total Cost:     {sol['total_cost']:.2f}\n")
                f.write("-" * 80 + "\n")
        
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
        print(f"  Check Cost:     {best_solution['check_cost']:.2f}")  # 新增检查成本
        print(f"  Fault Cost:     {best_solution['fault_cost']:.2f}")
        print(f"  Data Loss Cost: {best_solution['data_loss_cost']:.2f}")
        print(f"  Total Cost:     {best_solution['total_cost']:.2f}")
        print(f"  Budget: {best_solution['budget']}")
                
        # 打印执行时间
        print(f"\nPSO optimization completed! Time: {elapsed_time:.2f} seconds | Processes: {num_procs}")
        print(f"Feasible solutions found: {len(feasible_solutions)}")
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)