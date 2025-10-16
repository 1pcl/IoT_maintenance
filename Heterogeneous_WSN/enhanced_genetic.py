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
from deap import base, creator, tools, algorithms
import functools
import os
import json
from scene_generator import SceneGenerator

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

# ==================== 通用函数 ====================

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
            "check_cost": check_cost,
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
            param_keys = list(DEFAULT_PARAM_VALUES[zone].keys())
            param_values = [
                sol_info[zone]['preventive_check_days'],
                sol_info[zone]['frequency_heartbeat'] if sol_info[zone]['frequency_heartbeat'] is not None else DEFAULT_PARAM_VALUES[zone]["frequency_heartbeat"],
                sol_info[zone]['heartbeat_loss_threshold'] if sol_info[zone]['heartbeat_loss_threshold'] is not None else DEFAULT_PARAM_VALUES[zone]["heartbeat_loss_threshold"]
            ]
            param_deviation = sum(abs(p - DEFAULT_PARAM_VALUES[zone][k]) for p, k in zip(param_values, param_keys))
            total_param_deviation += param_deviation

    return (sol_info['total_cost'], total_module_cost, total_param_deviation)

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
    
    diversity = 0.0
    count = 0
    
    # 计算所有个体两两之间的汉明距离（仅模块部分）
    for i, j in itertools.combinations(range(len(population)), 2):
        # 为每个区域计算模块部分的汉明距离
        hamming_dist = 0
        for zone in ZONES:
            zone_modules = list(MODULES[zone].keys())
            n_modules = len(zone_modules)
            zone_start_idx = ZONES.index(zone) * (n_modules + 3)
            
            ind1 = population[i][zone_start_idx:zone_start_idx + n_modules]
            ind2 = population[j][zone_start_idx:zone_start_idx + n_modules]
            # 计算汉明距离（不同基因的数量）
            hamming_dist += sum(g1 != g2 for g1, g2 in zip(ind1, ind2))
        
        diversity += hamming_dist
        count += 1
    
    # 返回平均汉明距离
    return diversity / count if count > 0 else 0.0

# ==================== 遗传算法部分 ====================

def genetic_algorithm_optimization(pop_size=100, ngen=100, num_processes=100):
    """遗传算法优化 - 多区域版本"""
    # 初始化工具箱
    toolbox = base.Toolbox()
    
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
    
    # 自定义变异算子 - 增强变异强度（多区域版本）
    def mutMixed(individual, indpb, stagnation_count):
        """自定义变异操作，处理多区域混合类型基因"""
        n = len(individual)
        
        # 基于停滞计数增加变异强度
        mutation_intensity = 1.0 + stagnation_count * 0.3
        
        # 为每个区域处理变异
        current_idx = 0
        for zone in ZONES:
            zone_modules = list(MODULES[zone].keys())
            n_modules = len(zone_modules)
            
            # 模块部分：按位翻转（增强变异概率）
            for i in range(current_idx, current_idx + n_modules):
                # 动态调整变异概率
                adjusted_indpb = min(0.5, indpb * (1.0 + stagnation_count * 0.15))
                if random.random() < adjusted_indpb:
                    individual[i] = 1 - individual[i]
            
            # 参数部分：增强变异
            param_start = current_idx + n_modules
            param_indices = range(param_start, param_start + 3)
            for i in param_indices:
                # 动态调整变异概率
                adjusted_indpb = min(0.5, indpb * (1.0 + stagnation_count * 0.15))
                if random.random() < adjusted_indpb:
                    param_type = i - param_start
                    # 50%概率进行大范围变异（探索） - 增加探索比例
                    if random.random() < 0.5:
                        if param_type == 0:  # preventive_check_days
                            individual[i] = random.uniform(*param_ranges[zone]["preventive_check_days"])
                        elif param_type == 1:  # frequency_heartbeat
                            individual[i] = random.uniform(*param_ranges[zone]["frequency_heartbeat"])
                        elif param_type == 2:  # heartbeat_loss_threshold
                            individual[i] = random.uniform(*param_ranges[zone]["heartbeat_loss_threshold"])
                    # 50%概率进行大范围扰动（利用）
                    else:  
                        if param_type == 0:  # preventive_check_days
                            perturbation = random.randint(-15, 15) * mutation_intensity
                            individual[i] = max(param_ranges[zone]["preventive_check_days"][0], 
                                              min(param_ranges[zone]["preventive_check_days"][1], 
                                              individual[i] + perturbation))
                        elif param_type == 1:  # frequency_heartbeat
                            perturbation = random.uniform(0.5, 1.5)
                            new_val = individual[i] * perturbation
                            individual[i] = max(param_ranges[zone]["frequency_heartbeat"][0], 
                                              min(param_ranges[zone]["frequency_heartbeat"][1], 
                                              new_val))
                        elif param_type == 2:  # heartbeat_loss_threshold
                            perturbation = random.randint(-3, 3) * mutation_intensity
                            individual[i] = max(param_ranges[zone]["heartbeat_loss_threshold"][0], 
                                              min(param_ranges[zone]["heartbeat_loss_threshold"][1], 
                                              individual[i] + perturbation))
            
            current_idx += n_modules + 3
        
        return individual,
    
    # 设置种群规模和迭代次数
    pop = toolbox.population(n=pop_size)
    
    # 创建进程池
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
    hof = tools.HallOfFame(20)  # 增加精英保留数量
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [],
        'avg_total_cost': [],  # 平均值（只考虑可行解）
        'feasible_count': [],
        'mutation_rate': [],
        'genetic_diversity': []
    }
    
    # 记录历史可行解
    feasible_solutions = []
    for ind in pop:
        if ind.fitness.values[0] != float('inf'):
            feasible_solutions.append({
                "result": ind.result,
                "cost": ind.fitness.values[0]
            })
    
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
    stagnation_threshold = 15  # 适当增加停滞阈值
    improvement_threshold = 0.001  # 改进阈值，用于早停
    
    # 记录历史最佳成本
    historical_best = min_cost
    no_improvement_count = 0
    max_no_improvement = 20  # 最大无改进代数
    
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
                
                # 更新可行解列表
                if fitness_value[0] != float('inf'):
                    feasible_solutions.append({
                        "result": res,
                        "cost": fitness_value[0]
                    })
        
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
        if current_min < min_cost - improvement_threshold:
            stagnation_count = 0
            min_cost = current_min
        else:
            stagnation_count += 1
        
        # 检查是否有显著改进
        if current_min < historical_best - improvement_threshold:
            historical_best = current_min
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        convergence_data['min_total_cost'].append(current_min)
        convergence_data['avg_total_cost'].append(current_avg)
        convergence_data['feasible_count'].append(feasible_count)
        convergence_data['genetic_diversity'].append(calculate_genetic_diversity(pop))
        
        # 打印进度
        print(f"Gen {gen}/{ngen}: Min Cost={current_min:.2f}, Avg Cost={current_avg:.2f}, "
              f"Feasible={feasible_count}/{pop_size}, "
              f"GeneticDiv={convergence_data['genetic_diversity'][-1]:.1f}, "
              f"MutPb={mutpb:.3f}, "
              f"Stagnation={stagnation_count}/{stagnation_threshold}, "
              f"NoImprove={no_improvement_count}/{max_no_improvement}")
        
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
        
        # 早停机制 - 基于无改进代数
        if no_improvement_count >= max_no_improvement:
            print(f"Early stopping at generation {gen} due to no significant improvement for {max_no_improvement} generations.")
            break
        
        # 早停机制 - 基于停滞
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
    
    return best_individual.result, feasible_solutions, elapsed, num_processes, convergence_data

# ==================== 绘图函数 ====================

def plot_ga_convergence(convergence_data, elapsed_time, num_procs):
    """绘制遗传算法的收敛图"""
    # 确保目录存在 
    os.makedirs("results1", exist_ok=True)
    
    # 保存绘图数据到JSON文件
    plot_data = {
        "min_cost": convergence_data['min_total_cost'],
        "avg_cost": convergence_data['avg_total_cost'],
        "feasible_count": convergence_data['feasible_count'],
        "diversity": convergence_data['genetic_diversity'],
        "mutation_rate": convergence_data['mutation_rate'],
        "elapsed_time": elapsed_time,
        "num_procs": num_procs
    }
    
    with open("results1/en_ga_convergence_data.json", "w") as f:
        json.dump(plot_data, f, indent=4)
    
    # 1. 收敛曲线（最小值和平均值）
    plt.figure(figsize=(12, 8))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})
    
    generations = len(convergence_data['min_total_cost'])
    x = range(generations)
    
    plt.plot(x, convergence_data['min_total_cost'], 'b-', label='Min Cost', linewidth=2)
    plt.plot(x, convergence_data['avg_total_cost'], 'r--', label='Avg Cost (Feasible)', linewidth=2)
    
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Total Cost', fontsize=16)
    plt.title(f'Genetic Algorithm Optimization Convergence\n(Time: {elapsed_time:.2f}s, Processes: {num_procs})', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results1/en_ga_convergence.png', dpi=300)
    print("Saved: results1/en_ga_convergence.png")
    plt.close()
    
    # 2. 可行解数量变化
    plt.figure(figsize=(12, 8))
    
    plt.plot(x, convergence_data['feasible_count'], 'g-', label='Feasible Solutions', linewidth=2)
    
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Feasible Solutions Count', fontsize=16)
    plt.title('Feasible Solutions Evolution', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results1/en_ga_feasible_count.png', dpi=300)
    print("Saved: results1/en_ga_feasible_count.png")
    plt.close()
    
    # 3. 多样性变化
    plt.figure(figsize=(12, 8))
    
    plt.plot(x, convergence_data['genetic_diversity'], 'c-', label='Genetic Diversity', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Genetic Diversity', fontsize=16)
    plt.title('Genetic Diversity Evolution', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results1/en_ga_diversity.png', dpi=300)
    print("Saved: results1/en_ga_diversity.png")
    plt.close()
    
    # 4. 变异率变化
    plt.figure(figsize=(12, 8))
    
    plt.plot(x, convergence_data['mutation_rate'], 'm-', label='Mutation Rate', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Mutation Probability', fontsize=16)
    plt.title('Adaptive Mutation Rate', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results1/en_ga_mutation_rate.png', dpi=300)
    print("Saved: results1/en_ga_mutation_rate.png")
    plt.close()

def replot_from_saved_data():
    """从保存的数据重新绘制图表"""
    try:
        with open("results1/en_ga_convergence_data.json", "r") as f:
            plot_data = json.load(f)
        
        convergence_data = {
            'min_total_cost': plot_data["min_cost"],
            'avg_total_cost': plot_data["avg_cost"],
            'feasible_count': plot_data["feasible_count"],
            'genetic_diversity': plot_data["diversity"],
            'mutation_rate': plot_data["mutation_rate"]
        }
        
        plot_ga_convergence(convergence_data, plot_data["elapsed_time"], plot_data["num_procs"])
        print("Successfully replotted from saved data")
    except FileNotFoundError:
        print("No saved data found. Please run the optimization first.")
    except Exception as e:
        print(f"Error replotting: {e}")

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 进程数 
    num_processes = 100
    
    # 检查是否只需要重新绘图
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        replot_from_saved_data()
        sys.exit(0)
    
    print("="*80)
    print(f"Starting Genetic Algorithm Optimization with {num_processes} processes")
    print("="*80)
    
    try:
        # 运行遗传算法优化
        start_time = time.time()
        best_result, feasible_solutions, elapsed_time, num_procs, convergence_data = genetic_algorithm_optimization(
            pop_size=100,  # 种群规模
            ngen=100,      # 迭代次数
            num_processes=num_processes
        )
        total_time = time.time() - start_time
        
        # 绘制收敛图
        plot_ga_convergence(convergence_data, total_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_result)
        
        # 创建结果目录
        os.makedirs("results1", exist_ok=True)
        
        # 写入结果文件
        with open("results1/en_ga_optimization_results.txt", "w") as f:
            f.write("Genetic Algorithm Optimization Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Optimization Time: {total_time:.2f} seconds\n")
            f.write(f"Processes Used: {num_procs}\n")
            f.write(f"Budget: {config['budget']}\n")
            f.write(f"Best Total Cost: {best_solution['total_cost']:.2f}\n\n")
            
            # 最佳解决方案
            f.write("Optimal Configuration:\n")
            f.write("=" * 80 + "\n")
            
            # 为每个区域输出模块和参数
            for zone in ZONES:
                f.write(f"\n{zone.upper()}:\n")
                f.write(f"  Selected Modules: {best_solution['modules'][zone]}\n")
                f.write(f"  Preventive Check Days: {best_solution[zone]['preventive_check_days']} days\n")
                
                if best_solution[zone]['frequency_heartbeat'] is not None:
                    f.write(f"  Heartbeat Frequency: {best_solution[zone]['frequency_heartbeat']} seconds\n")
                    f.write(f"  Heartbeat Loss Threshold: {best_solution[zone]['heartbeat_loss_threshold']}\n")
            
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
                for i, sol in enumerate(solutions_info[:10]):  # 显示前10个
                    f.write(f"\nSolution #{i+1} (Total Cost: {sol['total_cost']:.2f}):\n")
                    for zone in ZONES:
                        f.write(f"  {zone.upper()}:\n")
                        f.write(f"    Modules: {sol['modules'][zone]}\n")
                        f.write(f"    Preventive Check Days: {sol[zone]['preventive_check_days']} days\n")
                        if sol[zone]['frequency_heartbeat'] is not None:
                            f.write(f"    Heartbeat Frequency: {sol[zone]['frequency_heartbeat']} seconds\n")
                            f.write(f"    Heartbeat Loss Threshold: {sol[zone]['heartbeat_loss_threshold']}\n")
                    f.write(f"  Base Cost:      {sol['base_cost']:.2f}\n")
                    f.write(f"  Module Cost:    {sol['module_cost']:.2f}\n")
                    f.write(f"  Check Cost:     {sol['check_cost']:.2f}\n")
                    f.write(f"  Fault Cost:     {sol['fault_cost']:.2f}\n")
                    f.write(f"  Data Loss Cost: {sol['data_loss_cost']:.2f}\n")
            else:
                f.write("No alternative feasible solutions found.\n")
        
        # 打印最佳解决方案
        print("\n" + "="*80)
        print("Optimal Configuration Found:")
        print("="*80)
        for zone in ZONES:
            print(f"\n{zone.upper()}:")
            print(f"  Selected Modules: {best_solution['modules'][zone]}")
            print(f"  Preventive Check Days: {best_solution[zone]['preventive_check_days']} days")
            
            if best_solution[zone]['frequency_heartbeat'] is not None:
                print(f"  Heartbeat Frequency: {best_solution[zone]['frequency_heartbeat']} seconds")
                print(f"  Heartbeat Loss Threshold: {best_solution[zone]['heartbeat_loss_threshold']}")
        
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
        print(f"Genetic algorithm optimization completed in {total_time:.2f} seconds")
        print(f"Using {num_procs} processes")
        print(f"Feasible solutions found: {len(feasible_solutions)}")
        print("Results saved to 'results1' directory")
        print("="*80)
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)