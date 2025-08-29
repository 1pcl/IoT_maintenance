# -*- coding: utf-8 -*-
from simulator import WSNSimulation
from module_cost import set_modules_cost
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
from deap import base, creator, tools, algorithms
import functools
import os
import json  # 添加json库用于保存数据
from scene_generator import SceneGenerator

# 使用非交互式后端避免GUI问题
matplotlib.use('Agg')

# 获取所有模块名称
all_modules = list(MODULES.keys())
print(f"Total modules available: {len(all_modules)}")
application = "animal_room"  #  animal_room, electricity_meter
scene_data = SceneGenerator.create_scene(application, visualize=True)
config=scene_data["config"]
# 设置模块成本
set_modules_cost(config)

# 定义适应度函数（单目标：最小化总成本）
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 单目标最小化

# 定义个体（二进制列表表示模块是否被选择 + 仿真参数）
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

# ==================== 通用函数 ====================

def evaluate_individual(individual):
    """评估个体的适应度，返回完整结果字典"""
    n_modules = len(all_modules)
    # 提取模块选择部分
    selected_modules = [all_modules[i] for i in range(n_modules) if individual[i] == 1]
    
    # 提取仿真参数部分
    param_start_idx = n_modules
    simulation_params = {
        "warning_energy": individual[param_start_idx],
        "preventive_check_days": int(round(individual[param_start_idx + 1])) if individual[param_start_idx + 1] > 0 else 1,
        "frequency_heartbeat": max(60, int(round(individual[param_start_idx + 2]))) if "heartbeat" in selected_modules else None,
        "heartbeat_loss_threshold": int(round(individual[param_start_idx + 3])) if "heartbeat" in selected_modules else None
    }
    
    # 创建仿真实例
    sim = WSNSimulation(scene_data, selected_modules, simulation_params)
    
    # 计算总成本
    total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost = calculate_total_cost_with_simulation(
        sim, selected_modules,config
    )
    
    # 计算预算超出部分
    budget = config["budget"]
    total_budget_cost = base_cost + module_cost + check_cost
    budget_exceed = max(0, total_budget_cost - budget)
    
    # 返回完整结果字典
    result = {
        "fitness": (float('inf'),) if budget_exceed > 0 else (total_cost,),
        "total_cost": total_cost,
        "base_cost": base_cost,
        "module_cost": module_cost,
        "check_cost": check_cost,
        "fault_cost": fault_cost,
        "data_loss_cost": data_loss_cost,
        "feasible": budget_exceed <= 0,
        "individual": individual,  # 保留原始个体
        "selected_modules": selected_modules,
        "simulation_params": simulation_params
    }
    
    return result

def solution_info_sort_key(sol_info):
    """解决方案信息排序函数，直接使用solution_info字典"""
    # 计算模块代价总和
    module_cost = sum(MODULES[m]['cost'] for m in sol_info['modules'])
    
    # 参数偏离度计算
    param_keys = list(DEFAULT_PARAM_VALUES.keys())
    param_values = [
        sol_info['warning_energy'],
        sol_info['preventive_check_days'],
        sol_info['frequency_heartbeat'] if sol_info['frequency_heartbeat'] is not None else DEFAULT_PARAM_VALUES["frequency_heartbeat"],
        sol_info['heartbeat_loss_threshold'] if sol_info['heartbeat_loss_threshold'] is not None else DEFAULT_PARAM_VALUES["heartbeat_loss_threshold"]
    ]
    param_deviation = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param_values, param_keys))

    return (sol_info['total_cost'], module_cost, param_deviation)

def extract_solution_info(solution_data):
    """从解决方案数据中提取信息"""
    return {
        "modules": solution_data["selected_modules"],
        "warning_energy": solution_data["simulation_params"]["warning_energy"],
        "preventive_check_days": solution_data["simulation_params"]["preventive_check_days"],
        "frequency_heartbeat": solution_data["simulation_params"]["frequency_heartbeat"],
        "heartbeat_loss_threshold": solution_data["simulation_params"]["heartbeat_loss_threshold"],
        "total_cost": solution_data["total_cost"],
        "base_cost": solution_data["base_cost"],
        "module_cost": solution_data["module_cost"],
        "check_cost": solution_data["check_cost"],
        "fault_cost": solution_data["fault_cost"],
        "data_loss_cost": solution_data["data_loss_cost"],
        "budget": config["budget"],
        "encoded_solution": solution_data["individual"]  # 原始个体
    }

def calculate_genetic_diversity(population):
    """计算种群的基因多样性（基于汉明距离）"""
    if len(population) < 2:
        return 0.0
    
    n_modules = len(all_modules)
    diversity = 0.0
    count = 0
    
    # 计算所有个体两两之间的汉明距离（仅模块部分）
    for i, j in itertools.combinations(range(len(population)), 2):
        ind1 = population[i][:n_modules]
        ind2 = population[j][:n_modules]
        # 计算汉明距离（不同基因的数量）
        hamming_dist = sum(g1 != g2 for g1, g2 in zip(ind1, ind2))
        diversity += hamming_dist
        count += 1
    
    # 返回平均汉明距离
    return diversity / count if count > 0 else 0.0

# ==================== 遗传算法部分 ====================

def genetic_algorithm_optimization(pop_size=150, ngen=50, num_processes=128):
    """遗传算法优化 - 评估次数减半版本"""
    # 初始化工具箱
    toolbox = base.Toolbox()
    n_modules = len(all_modules)
    
    # 定义参数范围
    warning_energy_range = (0.0, 50.0)
    check_day_min = round(config["frequency_sampling"]/(60*60*24))
    if check_day_min <= 0:
        check_day_min = 1        
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = config["frequency_sampling"]
    frequency_heartbeat_min = max(1, frequency_heartbeat_max/60)
    frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
    heartbeat_loss_threshold_range = (3, 15)
    
    # 个体生成函数
    def create_individual():
        # 模块选择部分（二进制）
        modules_part = [random.randint(0, 1) for _ in range(n_modules)]
        
        # 仿真参数部分
        params_part = [
            random.uniform(*warning_energy_range),
            random.uniform(*preventive_check_days_range),
            random.uniform(*frequency_heartbeat_range),
            random.uniform(*heartbeat_loss_threshold_range)
        ]
        
        return creator.Individual(modules_part + params_part)
    
    # 注册个体和种群生成器
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 注册评估函数
    toolbox.register("evaluate", evaluate_individual)
    
    # 注册选择算子（锦标赛选择）- 增加选择压力
    toolbox.register("select", tools.selTournament, tournsize=5)
    
    # 注册多种交叉算子
    toolbox.register("mate_two_point", tools.cxTwoPoint)
    toolbox.register("mate_uniform", tools.cxUniform, indpb=0.5)
    
    # 自定义变异算子 - 增强变异强度
    def mutMixed(individual, indpb, stagnation_count):
        """自定义变异操作，处理混合类型基因"""
        n = len(individual)
        
        # 基于停滞计数增加变异强度
        mutation_intensity = 1.0 + stagnation_count * 0.3
        
        # 模块部分：按位翻转（增强变异概率）
        for i in range(n_modules):
            # 动态调整变异概率
            adjusted_indpb = min(0.5, indpb * (1.0 + stagnation_count * 0.15))
            if random.random() < adjusted_indpb:
                individual[i] = 1 - individual[i]
        
        # 参数部分：增强变异
        param_indices = range(n_modules, n)
        for i in param_indices:
            # 动态调整变异概率
            adjusted_indpb = min(0.5, indpb * (1.0 + stagnation_count * 0.15))
            if random.random() < adjusted_indpb:
                # 50%概率进行大范围变异（探索） - 增加探索比例
                if random.random() < 0.5:
                    if i == n_modules:  # warning_energy
                        individual[i] = random.uniform(*warning_energy_range)
                    elif i == n_modules + 1:  # preventive_check_days
                        individual[i] = random.randint(*preventive_check_days_range)
                    elif i == n_modules + 2:  # frequency_heartbeat
                        individual[i] = random.uniform(*frequency_heartbeat_range)
                    elif i == n_modules + 3:  # heartbeat_loss_threshold
                        individual[i] = random.randint(*heartbeat_loss_threshold_range)
                # 50%概率进行大范围扰动（利用）
                else:  
                    if i == n_modules:  # warning_energy
                        perturbation = random.gauss(0, 10 * mutation_intensity)
                        individual[i] = max(warning_energy_range[0], 
                                          min(warning_energy_range[1], 
                                          individual[i] + perturbation))
                    elif i == n_modules + 1:  # preventive_check_days
                        perturbation = random.randint(-15, 15) * mutation_intensity
                        individual[i] = max(preventive_check_days_range[0], 
                                          min(preventive_check_days_range[1], 
                                          individual[i] + perturbation))
                    elif i == n_modules + 2:  # frequency_heartbeat
                        perturbation = random.uniform(0.5, 1.5)
                        new_val = individual[i] * perturbation
                        individual[i] = max(frequency_heartbeat_range[0], 
                                          min(frequency_heartbeat_range[1], 
                                          new_val))
                    elif i == n_modules + 3:  # heartbeat_loss_threshold
                        perturbation = random.randint(-3, 3) * mutation_intensity
                        individual[i] = max(heartbeat_loss_threshold_range[0], 
                                          min(heartbeat_loss_threshold_range[1], 
                                          individual[i] + perturbation))
        return individual,
    
    # 设置种群规模和迭代次数 - 减半
    pop = toolbox.population(n=pop_size)
    
    # 创建进程池 - 减少进程数
    print(f"Creating multiprocessing pool with {num_processes} workers for GA")
    pool = multiprocessing.Pool(processes=num_processes)
    toolbox.register("map", pool.map)
    
    # 评估初始种群
    print(f"Evaluating initial population (using {num_processes} processes)...")
    results = list(toolbox.map(toolbox.evaluate, pop))
    for ind, res in zip(pop, results):
        fitness_value = res["fitness"]
        ind.fitness.values = fitness_value
        ind.result = res
    
    # 创建Hall of Fame记录历史最优解
    hof = tools.HallOfFame(10)
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [],
        'avg_total_cost': [],  # 平均值（只考虑可行解）
        'feasible_count': [],
        'mutation_rate': [],
        'genetic_diversity': []
    }
    
    # 记录初始代
    total_costs = [ind.fitness.values[0] for ind in pop]
    feasible_costs = [cost for cost in total_costs if cost != float('inf')]
    feasible_count = len(feasible_costs)
    min_cost = min(total_costs) if total_costs else float('inf')
    avg_cost = sum(feasible_costs) / feasible_count if feasible_count > 0 else float('inf')
    
    convergence_data['min_total_cost'].append(min_cost)
    convergence_data['avg_total_cost'].append(avg_cost)
    convergence_data['feasible_count'].append(feasible_count)
    convergence_data['mutation_rate'].append(0.25)
    convergence_data['genetic_diversity'].append(calculate_genetic_diversity(pop))
    
    # 运行遗传算法
    print("Starting GA optimization...")
    start_time = time.time()
    
    # 遗传算法参数 - 增加选择压力和变异强度
    cxpb = 0.85
    base_mutpb = 0.3  # 增加基础变异概率
    min_mutpb = 0.2
    max_mutpb = 0.5
    stagnation_count = 0
    stagnation_threshold = 10  # 停滞阈值
    
    for gen in range(1, ngen + 1):
        # 自适应调整变异概率（基于多样性和停滞情况）
        current_genetic_diversity = calculate_genetic_diversity(pop)
        if current_genetic_diversity < 0.5 * convergence_data['genetic_diversity'][0] or stagnation_count > 0:
            mutpb = min(max_mutpb, base_mutpb * (1.5 + stagnation_count * 0.25))
        else:
            mutpb = base_mutpb
            
        convergence_data['mutation_rate'].append(mutpb)
        
        # 选择父代个体进行繁殖 - 增加选择压力
        parents = toolbox.select(pop, len(pop))
        
        # 复制选中的个体
        offspring = [toolbox.clone(ind) for ind in parents]
        
        # 应用交叉操作
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                if random.random() < 0.5:
                    toolbox.mate_two_point(child1, child2)
                else:
                    toolbox.mate_uniform(child1, child2)
                
                # 交叉后增加小变异
                extra_mut_prob = 0.3 + stagnation_count * 0.08
                if random.random() < extra_mut_prob:
                    mutMixed(child1, 0.3, stagnation_count)
                    del child1.fitness.values
                    if hasattr(child1, 'cost_breakdown'):
                        del child1.cost_breakdown
                if random.random() < extra_mut_prob:
                    mutMixed(child2, 0.3, stagnation_count)
                    del child2.fitness.values
                    if hasattr(child2, 'cost_breakdown'):
                        del child2.cost_breakdown
                else:
                    del child1.fitness.values
                    del child2.fitness.values
                    if hasattr(child1, 'cost_breakdown'):
                        del child1.cost_breakdown
                    if hasattr(child2, 'cost_breakdown'):
                        del child2.cost_breakdown
        
        # 应用变异操作
        for mutant in offspring:
            if random.random() < mutpb:
                mutMixed(mutant, mutpb, stagnation_count)
                del mutant.fitness.values
                if hasattr(mutant, 'cost_breakdown'):
                    del mutant.cost_breakdown
        
        # 评估所有新个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            results = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, res in zip(invalid_ind, results):
                fitness_value = res["fitness"]
                ind.fitness.values = fitness_value
                ind.result = res
        
        # 合并父代和子代
        combined_pop = pop + offspring
        
        # 动态平衡选择策略 - 增加精英比例
        elite_ratio = max(0.6, 0.8 - stagnation_count * 0.03)
        elite_size = int(elite_ratio * pop_size)
        
        # 按适应度排序
        sorted_pop = sorted(combined_pop, key=lambda ind: ind.fitness.values[0])
        elite = sorted_pop[:elite_size]
        
        # 选择非精英部分（保持多样性）
        non_elite = sorted_pop[elite_size:]
        random.shuffle(non_elite)
        
        # 构建新一代种群
        pop[:] = elite + non_elite[:pop_size - elite_size]
        
        # 更新Hall of Fame
        hof.update(pop)
        
        # 记录当前代数据
        total_costs = [ind.fitness.values[0] for ind in pop]
        feasible_costs = [cost for cost in total_costs if cost != float('inf')]
        feasible_count = len(feasible_costs)
        current_min = min(total_costs) if total_costs else float('inf')
        current_avg = sum(feasible_costs) / feasible_count if feasible_count > 0 else float('inf')
        
        # 更新停滞计数器
        if current_min < min_cost - 0.001:
            stagnation_count = 0
            min_cost = current_min
        else:
            stagnation_count += 1
        
        convergence_data['min_total_cost'].append(current_min)
        convergence_data['avg_total_cost'].append(current_avg)
        convergence_data['feasible_count'].append(feasible_count)
        convergence_data['genetic_diversity'].append(calculate_genetic_diversity(pop))
        
        # 打印进度
        print(f"Gen {gen}/{ngen}: Min Cost={current_min:.2f}, Avg Cost={current_avg:.2f}, "
              f"Feasible={feasible_count}/{pop_size}, "
              f"GeneticDiv={convergence_data['genetic_diversity'][-1]:.1f}, "
              f"MutPb={mutpb:.3f}, "
              f"Stagnation={stagnation_count}/{stagnation_threshold}")
        
        # 智能重启机制
        if stagnation_count > stagnation_threshold // 2 or convergence_data['genetic_diversity'][-1] < 0.3 * convergence_data['genetic_diversity'][0]:
            num_replace = max(10, int(0.3 * pop_size))
            pop.sort(key=lambda ind: ind.fitness.values[0])
            best_to_keep = tools.selBest(pop, pop_size - num_replace)
            new_individuals = toolbox.population(n=num_replace)
            results = toolbox.map(toolbox.evaluate, new_individuals)
            for ind, res in zip(new_individuals, results):
                fitness_value = res["fitness"]
                ind.fitness.values = fitness_value
                ind.result = res
            pop[:] = best_to_keep + new_individuals
            stagnation_count = 0
            print(f"Gen {gen}: Partial restart ({num_replace} new individuals)")
        
        # 早停机制
        if stagnation_count >= stagnation_threshold:
            print(f"Early stopping at generation {gen} due to convergence.")
            break
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 从Hall of Fame获取最佳个体
    feasible_individuals = [ind for ind in hof if ind.fitness.values[0] != float('inf')]
    if feasible_individuals:
        best_individual = min(feasible_individuals, key=lambda ind: ind.fitness.values[0])
    else:
        best_individual = min(pop, key=lambda ind: ind.fitness.values[0])
    
    return best_individual.result, pop, elapsed, num_processes, convergence_data

# ==================== 模拟退火部分 ====================

def neighbor_function(current_solution, temperature, initial_temp, ga_diversity=1.0, stagnation_count=0):
    """生成邻域解，支持自适应扰动和停滞响应"""
    n_modules = len(all_modules)
    neighbor = creator.Individual(current_solution[:])
    
    # 定义参数范围
    warning_energy_range = (0.0, 50.0)
    check_day_min = round(config["frequency_sampling"]/(60*60*24))
    if check_day_min <= 0:
        check_day_min = 1        
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = config["frequency_sampling"]
    frequency_heartbeat_min = max(1, frequency_heartbeat_max/60)
    frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
    heartbeat_loss_threshold_range = (3, 15)
    
    # 温度影响因子
    perturbation_factor = max(0.1, min(2.0, temperature / initial_temp))
    
    # 基于停滞计数增加扰动强度
    stagnation_factor = 1.0 + stagnation_count * 0.3
    
    # 结合GA多样性和停滞因子调整扰动强度
    perturbation_factor = perturbation_factor * ga_diversity * stagnation_factor
    
    # 随机选择扰动类型
    perturbation_type = random.choice([
        "flip_module", 
        "adjust_warning_energy",
        "adjust_preventive_days",
        "adjust_heartbeat_freq",
        "adjust_heartbeat_threshold"
    ])
    
    # 模块部分：翻转一个随机模块
    if perturbation_type == "flip_module":
        num_flips = min(n_modules, 1 + int(stagnation_count * 0.2))
        for _ in range(num_flips):
            idx = random.randint(0, n_modules - 1)
            neighbor[idx] = 1 - neighbor[idx]
    
    # 参数部分：调整warning_energy
    elif perturbation_type == "adjust_warning_energy":
        param_idx = n_modules
        std_dev = 8 * perturbation_factor
        perturbation = random.gauss(0, std_dev)
        neighbor[param_idx] = max(warning_energy_range[0], 
                                 min(warning_energy_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    # 参数部分：调整preventive_check_days
    elif perturbation_type == "adjust_preventive_days":
        param_idx = n_modules + 1
        perturbation_range = int(15 * perturbation_factor)
        perturbation = random.randint(-perturbation_range, perturbation_range)
        neighbor[param_idx] = max(preventive_check_days_range[0], 
                                 min(preventive_check_days_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    # 参数部分：调整frequency_heartbeat
    elif perturbation_type == "adjust_heartbeat_freq":
        param_idx = n_modules + 2
        min_factor = max(0.5, 0.7 - stagnation_count * 0.05)
        max_factor = min(1.5, 1.3 + stagnation_count * 0.05)
        perturbation = random.uniform(min_factor, max_factor)
        new_val = neighbor[param_idx] * perturbation
        neighbor[param_idx] = max(frequency_heartbeat_range[0], 
                                 min(frequency_heartbeat_range[1], 
                                 new_val))
    
    # 参数部分：调整heartbeat_loss_threshold
    elif perturbation_type == "adjust_heartbeat_threshold":
        param_idx = n_modules + 3
        perturbation_range = int(4 * perturbation_factor)
        perturbation = random.randint(-perturbation_range, perturbation_range)
        neighbor[param_idx] = max(heartbeat_loss_threshold_range[0], 
                                 min(heartbeat_loss_threshold_range[1], 
                                 neighbor[param_idx] + perturbation))
    
    return neighbor

def simulated_annealing_optimization(initial_solution=None, initial_result=None, num_processes=128, max_iter=50, ga_diversity=1.0):
    """模拟退火优化 - 使用GA评估结果作为初始解"""
    # 初始化工具箱
    n_modules = len(all_modules)
    
    # 定义参数范围
    warning_energy_range = (0.0, 50.0)
    check_day_min = round(config["frequency_sampling"]/(60*60*24))
    if check_day_min <= 0:
        check_day_min = 1        
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = config["frequency_sampling"]
    frequency_heartbeat_min = max(1, frequency_heartbeat_max/60)
    frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
    heartbeat_loss_threshold_range = (3, 15)
    
    # 个体生成函数
    def create_individual():
        # 模块选择部分（二进制）
        modules_part = [random.randint(0, 1) for _ in range(n_modules)]
        
        # 仿真参数部分
        params_part = [
            random.uniform(*warning_energy_range),
            random.uniform(*preventive_check_days_range),
            random.uniform(*frequency_heartbeat_range),
            random.uniform(*heartbeat_loss_threshold_range)
        ]
        
        return creator.Individual(modules_part + params_part)
    
    # 模拟退火参数 - 优化温度调度
    initial_temp = 1500.0  # 提高初始温度
    final_temp = 0.1
    cooling_rate = 0.90  # 降低冷却率
    max_stagnation = 20  # 减少最大停滞次数
    
    # 创建初始解 - 使用GA提供的解和评估结果
    if initial_solution is not None and initial_result is not None:
        current_solution = creator.Individual(initial_solution[:])
        current_result = initial_result
        current_cost = current_result["fitness"][0]
        print(f"SA using GA result with cost: {current_cost:.2f}")
    elif initial_solution is not None:
        current_solution = creator.Individual(initial_solution[:])
        current_result = evaluate_individual(current_solution)
        current_cost = current_result["fitness"][0]
        print(f"SA evaluated initial solution cost: {current_cost:.2f}")
    else:
        current_solution = create_individual()
        current_result = evaluate_individual(current_solution)
        current_cost = current_result["fitness"][0]
        print(f"SA random initial solution cost: {current_cost:.2f}")
    
    # 记录当前解和最佳解
    best_solution = creator.Individual(current_solution[:])
    best_cost = current_cost
    best_result = current_result
    
    # +++ 添加全局精英保护 +++
    global_best_solution = creator.Individual(current_solution[:])
    global_best_cost = current_cost
    global_best_temp = initial_temp
    
    # 初始的平均值
    initial_avg = current_cost if current_result["feasible"] else float('inf')
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [best_cost],
        'avg_total_cost': [initial_avg],  # 平均值（只考虑可行解）
        'current_cost': [current_cost],
        'temperature': [initial_temp],
        'feasible_count': [1 if current_cost != float('inf') else 0]
    }
    
    # 记录历史可行解
    feasible_solutions = []
    if current_result["feasible"]:
        feasible_solutions.append({
            "result": current_result,
            "cost": current_cost
        })
    
    print("Starting simulated annealing optimization...")
    start_time = time.time()
    
    # 初始温度
    temperature = initial_temp
    iteration = 0
    stagnation_count = 0
    
    # 创建进程池 - 减少进程数
    print(f"Creating multiprocessing pool with {num_processes} workers for SA")
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 定义批量生成邻域解的函数
    def generate_neighbors(current, neighbors_per_iteration):
        return [neighbor_function(current, temperature, initial_temp, ga_diversity, stagnation_count) 
                for _ in range(neighbors_per_iteration)]
    
    # 主循环
    while temperature > final_temp and iteration < max_iter and stagnation_count < max_stagnation:
        # 一次生成多个邻域解
        neighbors = generate_neighbors(current_solution, num_processes)
        
        # 并行评估所有邻域解
        results = pool.map(evaluate_individual, neighbors)
        
        # 找出最佳邻域解并计算平均值
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_neighbor_result = None
        feasible_costs = []  # 存储可行解的成本
        
        for neighbor, result in zip(neighbors, results):
            neighbor_cost = result["fitness"][0]
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
                best_neighbor_result = result
            
            # 记录可行解的成本
            if result["feasible"]:
                feasible_costs.append(neighbor_cost)
        
        # 计算平均值（只考虑可行解）
        avg_cost = sum(feasible_costs) / len(feasible_costs) if feasible_costs else float('inf')
        
        # 更新可行解列表
        if best_neighbor_result["feasible"]:
            feasible_solutions.append({
                "result": best_neighbor_result,
                "cost": best_neighbor_cost
            })
        
        # 计算成本差
        cost_difference = best_neighbor_cost - current_cost
        
        # Metropolis准则：决定是否接受新解
        accept = False
        if cost_difference < 0:
            accept = True
        elif temperature > 0:
            acceptance_prob = math.exp(-cost_difference / temperature)
            if random.random() < acceptance_prob:
                accept = True
        
        # 如果接受新解，更新当前状态
        if accept:
            current_solution = creator.Individual(best_neighbor[:])
            current_cost = best_neighbor_cost
            current_result = best_neighbor_result
            
            # 更新最佳解
            if current_cost < best_cost:
                best_solution = creator.Individual(best_neighbor[:])
                best_cost = current_cost
                best_result = best_neighbor_result
                stagnation_count = 0
                print(f"Iter {iteration}: New best cost = {best_cost:.2f}, Temp = {temperature:.2f}")
                
            # 更新全局最优解（精英保护）
            if current_cost < global_best_cost:
                global_best_solution = creator.Individual(best_neighbor[:])
                global_best_cost = current_cost
                global_best_temp = temperature
                print(f"Iter {iteration}: New GLOBAL best cost = {global_best_cost:.2f}")
                
        else:
            stagnation_count += 1
        
        # 重新加热机制：当停滞计数达到阈值时增加温度
        if stagnation_count >= max_stagnation//2:
            # 使用全局最优温度进行重新加热
            new_temperature = global_best_temp * 1.2
            
            # 设置温度上限为初始温度的80%
            if new_temperature > initial_temp * 0.8:
                new_temperature = initial_temp * 0.8
            
            # 确保不低于当前温度
            new_temperature = max(new_temperature, temperature * 1.1)
            
            temperature = new_temperature
            stagnation_count = 0
            print(f"Iter {iteration}: Strategic reheating to {temperature:.2f} (based on global best temp {global_best_temp:.2f})")
        
        # 记录收敛数据
        convergence_data['min_total_cost'].append(global_best_cost)  # 使用全局最优解
        convergence_data['avg_total_cost'].append(avg_cost)  # 记录平均值
        convergence_data['current_cost'].append(current_cost)
        convergence_data['temperature'].append(temperature)
        convergence_data['feasible_count'].append(len(feasible_solutions))
        
        # 降温
        temperature *= cooling_rate
        iteration += 1
        
        # 每10次迭代打印进度
        if iteration % 10 == 0:
            print(f"Iter {iteration}: Current cost = {current_cost:.2f}, "
                  f"Best cost = {best_cost:.2f}, "
                  f"Global best = {global_best_cost:.2f}, "
                  f"Avg cost = {avg_cost:.2f}, "
                  f"Temp = {temperature:.2f}, Stagnation = {stagnation_count}, "
                  f"Feasible = {len(feasible_solutions)}")
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # +++ 确保返回全局最优解 +++
    # 评估全局最优解
    global_best_result = evaluate_individual(global_best_solution)
    
    # 如果全局最优解不在可行解列表中，添加到列表中
    if global_best_result["feasible"]:
        found = False
        for sol in feasible_solutions:
            if sol["result"]["individual"] == global_best_solution:
                found = True
                break
        if not found:
            feasible_solutions.append({
                "result": global_best_result,
                "cost": global_best_cost
            })
    
    # 从可行解中选择最佳解
    if feasible_solutions:
        feasible_solutions.sort(key=lambda x: x["result"]["total_cost"])
        best_feasible = feasible_solutions[0]["result"]
    else:
        best_feasible = global_best_result
    
    # 填充收敛数据结构
    convergence_data_full = {
        'min_total_cost': convergence_data['min_total_cost'],
        'avg_total_cost': convergence_data['avg_total_cost'],  # 平均值
        'current_cost': convergence_data['current_cost'],
        'feasible_count': convergence_data['feasible_count'],
        'mutation_rate': [cooling_rate] * len(convergence_data['min_total_cost']),
        'genetic_diversity': [0] * len(convergence_data['min_total_cost']),
        'temperature': convergence_data['temperature']
    }
    
    print(f"SA completed in {iteration} iterations. Best cost: {best_feasible['total_cost']:.2f}")
    return best_feasible, feasible_solutions, elapsed, num_processes, convergence_data_full

# ==================== 混合优化部分 ====================

def hybrid_ga_sa_optimization(num_processes=128):
    """结合遗传算法和模拟退火的混合优化 - 总评估次数减半"""
    # 第一阶段：遗传算法全局搜索
    print("="*80)
    print("Starting Genetic Algorithm phase...")
    print("="*80)
    
    # GA阶段参数 - 评估次数减半
    ga_best, ga_population, ga_elapsed, ga_num_procs, ga_convergence = genetic_algorithm_optimization(
        pop_size=150,  # 种群规模
        ngen=50,       # 减少迭代次数
        num_processes=num_processes
    )
    
    # 计算GA的多样性
    ga_genetic_diversity = ga_convergence['genetic_diversity'][-1] if ga_convergence['genetic_diversity'] else 1.0
    print(f"GA phase completed. Best cost: {ga_best['total_cost']:.2f}, Diversity: {ga_genetic_diversity:.2f}")
    
    # 提取GA的最佳解作为SA的起点
    sa_start_solution = ga_best["individual"]
    
    # 第二阶段：模拟退火局部优化 - 使用GA的评估结果
    print("\n" + "="*80)
    print("Starting Simulated Annealing phase...")
    print("="*80)
    
    # SA阶段参数 - 评估次数减半
    sa_best_feasible, sa_feasible_solutions, sa_elapsed, sa_num_procs, sa_convergence = simulated_annealing_optimization(
        initial_solution=sa_start_solution,
        initial_result=ga_best,  # 使用GA的评估结果
        num_processes=num_processes,
        max_iter=50,  # 减少迭代次数
        ga_diversity=ga_genetic_diversity
    )
    
    # 合并结果
    elapsed_time = ga_elapsed + sa_elapsed
    convergence_data = {
        "ga_min_cost": ga_convergence["min_total_cost"],
        "ga_avg_cost": ga_convergence["avg_total_cost"],  # GA平均值
        "sa_min_cost": sa_convergence["min_total_cost"],
        "sa_avg_cost": sa_convergence["avg_total_cost"],  # SA平均值
        "ga_feasible": ga_convergence["feasible_count"],
        "sa_feasible": sa_convergence["feasible_count"],
        "ga_diversity": ga_convergence["genetic_diversity"]
    }
    
    return sa_best_feasible, sa_feasible_solutions, elapsed_time, num_processes, convergence_data

def plot_hybrid_convergence(convergence_data, elapsed_time, num_procs):
    """绘制混合优化的收敛图 - 确保GA到SA的平滑过渡，包括平均值"""
    # 确保目录存在
    os.makedirs("results", exist_ok=True)
    
    # 确保SA阶段第一点与GA最后一点相同
    if convergence_data["sa_min_cost"] and convergence_data["ga_min_cost"]:
        convergence_data["sa_min_cost"][0] = convergence_data["ga_min_cost"][-1]
        convergence_data["sa_avg_cost"][0] = convergence_data["ga_avg_cost"][-1]
    
    # 保存绘图数据到JSON文件
    plot_data = {
        "ga_min_cost": convergence_data["ga_min_cost"],
        "ga_avg_cost": convergence_data["ga_avg_cost"],
        "sa_min_cost": convergence_data["sa_min_cost"],
        "sa_avg_cost": convergence_data["sa_avg_cost"],
        "ga_feasible": convergence_data["ga_feasible"],
        "sa_feasible": convergence_data["sa_feasible"],
        "ga_diversity": convergence_data["ga_diversity"],
        "elapsed_time": elapsed_time,
        "num_procs": num_procs
    }
    
    with open("results/hybrid_convergence_data.json", "w") as f:
        json.dump(plot_data, f, indent=4)
    
    # 1. 组合收敛曲线（最小值和平均值）
    plt.figure(figsize=(12, 8))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})
    
    # GA阶段
    ga_generations = len(convergence_data["ga_min_cost"])
    ga_x = range(ga_generations)
    plt.plot(ga_x, convergence_data["ga_min_cost"], 'b-', label='GA Min Cost', linewidth=2)
    plt.plot(ga_x, convergence_data["ga_avg_cost"], 'b--', label='GA Avg Cost (Feasible)', linewidth=2)
    
    # SA阶段
    sa_iterations = len(convergence_data["sa_min_cost"])
    sa_x = range(ga_generations, ga_generations + sa_iterations)
    plt.plot(sa_x, convergence_data["sa_min_cost"], 'r-', label='SA Min Cost', linewidth=2)
    plt.plot(sa_x, convergence_data["sa_avg_cost"], 'r--', label='SA Avg Cost (Feasible)', linewidth=2)
    
    # 标记阶段转换
    plt.axvline(x=ga_generations-1, color='g', linestyle='--', linewidth=2, label='GA to SA Transition')
    
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Total Cost', fontsize=16)
    plt.title(f'Hybrid GA-SA Optimization Convergence (Evaluation Reduced)\n(Time: {elapsed_time:.2f}s, Processes: {num_procs})', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results/hybrid_convergence_reduced.png', dpi=300)
    print("Saved: results/hybrid_convergence_reduced.png")
    plt.close()
    
    # 2. 可行解数量变化
    plt.figure(figsize=(12, 8))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})
    
    # GA阶段
    plt.plot(ga_x, convergence_data["ga_feasible"], 'g-', label='GA Feasible', linewidth=2)
    
    # SA阶段
    plt.plot(sa_x, convergence_data["sa_feasible"], 'm-', label='SA Feasible', linewidth=2)
    
    plt.axvline(x=ga_generations-1, color='b', linestyle='--', linewidth=2, label='Phase Transition')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Feasible Solutions', fontsize=16)
    plt.title('Feasible Solutions Evolution (Evaluation Reduced)', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results/hybrid_feasible_count_reduced.png', dpi=300)
    print("Saved: results/hybrid_feasible_count_reduced.png")
    plt.close()
    
    # 3. 多样性变化
    if "ga_diversity" in convergence_data:
        plt.figure(figsize=(12, 8))
        
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 14})
        
        plt.plot(ga_x, convergence_data["ga_diversity"], 'c-', label='GA Diversity', linewidth=2)
        plt.axvline(x=ga_generations-1, color='r', linestyle='--', linewidth=2, label='Phase Transition')
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Genetic Diversity', fontsize=16)
        plt.title('Genetic Diversity in GA Phase (Evaluation Reduced)', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig('results/hybrid_diversity_reduced.png', dpi=300)
        print("Saved: results/hybrid_diversity_reduced.png")
        plt.close()

def replot_from_saved_data():
    """从保存的数据重新绘制图表"""
    try:
        with open("results/hybrid_convergence_data.json", "r") as f:
            plot_data = json.load(f)
        
        convergence_data = {
            "ga_min_cost": plot_data["ga_min_cost"],
            "ga_avg_cost": plot_data["ga_avg_cost"],
            "sa_min_cost": plot_data["sa_min_cost"],
            "sa_avg_cost": plot_data["sa_avg_cost"],
            "ga_feasible": plot_data["ga_feasible"],
            "sa_feasible": plot_data["sa_feasible"],
            "ga_diversity": plot_data["ga_diversity"]
        }
        
        plot_hybrid_convergence(convergence_data, plot_data["elapsed_time"], plot_data["num_procs"])
        print("Successfully replotted from saved data")
    except FileNotFoundError:
        print("No saved data found. Please run the optimization first.")
    except Exception as e:
        print(f"Error replotting: {e}")

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 进程数
    num_processes = 128
    
    # 检查是否只需要重新绘图
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        replot_from_saved_data()
        sys.exit(0)
    
    print("="*80)
    print(f"Starting Hybrid GA-SA Optimization with {num_processes} processes (Evaluation Reduced)")
    print(f"Application: {application}")
    print("="*80)
    
    try:
        # 运行混合优化
        start_time = time.time()
        best_feasible, feasible_solutions, elapsed_time, num_procs, convergence_data = hybrid_ga_sa_optimization(num_processes)
        total_time = time.time() - start_time
        
        # 绘制混合收敛图
        plot_hybrid_convergence(convergence_data, total_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_feasible)
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 写入结果文件
        with open("results/hybrid_optimization_results_reduced.txt", "w") as f:
            f.write("Hybrid GA-SA Optimization Results (Evaluation Reduced)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Optimization Time: {total_time:.2f} seconds\n")
            f.write(f"Processes Used: {num_procs}\n")
            f.write(f"Application: {application}\n")
            f.write(f"Budget: {config['budget']}\n")
            f.write(f"Best Total Cost: {best_solution['total_cost']:.2f}\n\n")
            
            # 最佳解决方案
            f.write("Optimal Configuration:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Selected Modules: {best_solution['modules']}\n\n")
            f.write("Global Parameters:\n")
            f.write(f"  warning_energy: {best_solution['warning_energy']:.2f}%\n")
            f.write(f"  preventive_check_days: {best_solution['preventive_check_days']} days\n")
            
            if best_solution['frequency_heartbeat'] is not None:
                f.write("\nModule Parameters (Heartbeat):\n")
                f.write(f"  frequency_heartbeat: {best_solution['frequency_heartbeat']} seconds\n")
                f.write(f"  heartbeat_loss_threshold: {best_solution['heartbeat_loss_threshold']}\n")
            
            f.write("\nCost Breakdown:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Base Cost:         {best_solution['base_cost']:.2f}\n")
            f.write(f"Module Cost:       {best_solution['module_cost']:.2f}\n")
            f.write(f"Check Cost:        {best_solution['check_cost']:.2f}\n")
            f.write(f"Fault Cost:        {best_solution['fault_cost']:.2f}\n")
            f.write(f"Data Loss Cost:    {best_solution['data_loss_cost']:.2f}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Cost:        {best_solution['total_cost']:.2f}\n\n")
            
            # 其他可行解
            if feasible_solutions:
                solutions_info = []
                for sol in feasible_solutions:
                    sol_info = extract_solution_info(sol["result"])
                    solutions_info.append(sol_info)
                
                solutions_info.sort(key=solution_info_sort_key)
                
                f.write(f"Alternative Feasible Solutions ({len(feasible_solutions)} found):\n")
                f.write("=" * 80 + "\n")
                for i, sol in enumerate(solutions_info[:5]):
                    f.write(f"\nSolution #{i+1} (Total Cost: {sol['total_cost']:.2f}):\n")
                    f.write(f"  Modules: {sol['modules']}\n")
                    f.write(f"  Warning Energy: {sol['warning_energy']:.2f}%\n")
                    f.write(f"  Preventive Check Days: {sol['preventive_check_days']} days\n")
                    if sol['frequency_heartbeat'] is not None:
                        f.write(f"  Heartbeat Frequency: {sol['frequency_heartbeat']} seconds\n")
                        f.write(f"  Heartbeat Loss Threshold: {sol['heartbeat_loss_threshold']}\n")
                    f.write(f"  Base Cost:      {sol['base_cost']:.2f}\n")
                    f.write(f"  Module Cost:    {sol['module_cost']:.2f}\n")
                    f.write(f"  Check Cost:     {sol['check_cost']:.2f}\n")
                    f.write(f"  Fault Cost:     {sol['fault_cost']:.2f}\n")
                    f.write(f"  Data Loss Cost: {sol['data_loss_cost']:.2f}\n")
            else:
                f.write("No alternative feasible solutions found.\n")
        
        # 打印最佳解决方案
        print("\n" + "="*80)
        print("Optimal Configuration Found (Evaluation Reduced):")
        print("="*80)
        print(f"Selected Modules: {best_solution['modules']}")
        print("\nGlobal Parameters:")
        print(f"  Warning Energy: {best_solution['warning_energy']:.2f}%")
        print(f"  Preventive Check Days: {best_solution['preventive_check_days']} days")
        
        if best_solution['frequency_heartbeat'] is not None:
            print("\nModule Parameters (Heartbeat):")
            print(f"  Frequency: {best_solution['frequency_heartbeat']} seconds")
            print(f"  Loss Threshold: {best_solution['heartbeat_loss_threshold']}")
        
        print("\nCost Breakdown:")
        print(f"  Base Cost:         {best_solution['base_cost']:.2f}")
        print(f"  Module Cost:       {best_solution['module_cost']:.2f}")
        print(f"  Check Cost:        {best_solution['check_cost']:.2f}")
        print(f"  Fault Cost:        {best_solution['fault_cost']:.2f}")
        print(f"  Data Loss Cost:    {best_solution['data_loss_cost']:.2f}")
        print("-"*40)
        print(f"Total Cost:        {best_solution['total_cost']:.2f}")
        print(f"Budget:            {best_solution['budget']}")
        
        # 打印执行时间
        print("\n" + "="*80)
        print(f"Hybrid optimization completed in {total_time:.2f} seconds (Evaluation Reduced)")
        print(f"Using {num_procs} processes")
        print(f"Feasible solutions found: {len(feasible_solutions) if feasible_solutions else 0}")
        print("Results saved to 'results' directory")
        print("="*80)
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)