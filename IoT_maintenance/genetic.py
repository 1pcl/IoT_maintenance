# -*- coding: utf-8 -*-
from simulator import WSNSimulation
from module_cost import set_modules_cost
from utilities import MODULES, DEFAULT_PARAM_VALUES
import time
from calculate_cost import calculate_total_cost_with_simulation
import random
from deap import base, creator, tools, algorithms
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import itertools
import functools
from scene_generator import SceneGenerator
import json
import os

# 使用非交互式后端避免GUI问题
matplotlib.use('Agg')

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
    n_modules = len(all_modules)
    selected_modules1 = [all_modules[i] for i in range(n_modules) if ind1[i] == 1]
    selected_modules2 = [all_modules[i] for i in range(n_modules) if ind2[i] == 1]

    module_cost1 = sum(MODULES[m]['cost'] for m in selected_modules1)
    module_cost2 = sum(MODULES[m]['cost'] for m in selected_modules2)

    if module_cost1 < module_cost2:
        return -1
    elif module_cost1 > module_cost2:
        return 1

    # 参数偏离程度
    param1 = ind1[n_modules:]
    param2 = ind2[n_modules:]

    param_keys = list(DEFAULT_PARAM_VALUES.keys())[1:]  
    deviation1 = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param1, param_keys))
    deviation2 = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param2, param_keys))

    if deviation1 < deviation2:
        return -1
    elif deviation1 > deviation2:
        return 1

    return 0  # 完全相等

def solution_sort_key(sol):
    # 计算模块代价总和
    module_cost = sum(MODULES[m]['cost'] for m in sol['modules'])

    # 参数偏离度计算
    param_keys = list(DEFAULT_PARAM_VALUES.keys())[1:]  
    param_values = [
        sol['preventive_check_days'],
        sol['frequency_heartbeat'] if sol['frequency_heartbeat'] is not None else DEFAULT_PARAM_VALUES["frequency_heartbeat"],
        sol['heartbeat_loss_threshold'] if sol['heartbeat_loss_threshold'] is not None else DEFAULT_PARAM_VALUES["heartbeat_loss_threshold"]
    ]
    param_deviation = sum(abs(p - DEFAULT_PARAM_VALUES[k]) for p, k in zip(param_values, param_keys))

    return (sol['total_cost'], module_cost, param_deviation)

# 获取所有模块名称
all_modules = list(MODULES.keys())
print(f"Total modules available: {len(all_modules)}")
application = "animal_room"  # animal_room, electricity_meter
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

def evaluate_individual(individual):
    """评估个体的适应度并返回适应度值和成本明细"""
    n_modules = len(all_modules)
    # 提取模块选择部分
    selected_modules = [all_modules[i] for i in range(n_modules) if individual[i] == 1]
    
    # 提取仿真参数部分
    param_start_idx = n_modules
    simulation_params = {
        "preventive_check_days": int(round(individual[param_start_idx])) if individual[param_start_idx] > 0 else 1,
        "frequency_heartbeat": max(60, int(round(individual[param_start_idx + 1]))) if "heartbeat" in selected_modules else None,
        "heartbeat_loss_threshold": int(round(individual[param_start_idx + 2])) if "heartbeat" in selected_modules else None
    }
    
    # 创建仿真实例
    sim = WSNSimulation(scene_data, selected_modules, simulation_params)
    
    # 计算总成本
    total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost = calculate_total_cost_with_simulation(
        sim, selected_modules,config
    )
    
    # 计算预算超出部分
    budget = config["budget"]
    total_budget_cost = total_cost
    budget_exceed = max(0, total_budget_cost - budget)

    # 创建成本明细字典
    cost_breakdown = {
        "total_cost": total_cost,
        "base_cost": base_cost,
        "module_cost": module_cost,
        "check_cost": check_cost,
        "fault_cost": fault_cost,
        "data_loss_cost": data_loss_cost,
    }
    
    
    # 保持硬约束，预算超支直接返回无穷大
    if budget_exceed > 0:
        return (float('inf'),), cost_breakdown
    
    return (total_cost,), cost_breakdown

def genetic_algorithm_optimization(num_processes=100):
    """执行遗传算法优化（预算约束下的单目标优化）"""
    # 初始化工具箱
    toolbox = base.Toolbox()
    n_modules = len(all_modules)
    
    check_day_min = round(config["frequency_sampling"] / (60 * 60 * 24))
    if check_day_min <= 0:
        check_day_min = 1
    preventive_check_days_range = (check_day_min, 180)
    frequency_heartbeat_max = config["frequency_sampling"]
    frequency_heartbeat_min=frequency_heartbeat_max/60
    if frequency_heartbeat_min<=0:
        frequency_heartbeat_min=1
    frequency_heartbeat_range = (frequency_heartbeat_min, frequency_heartbeat_max)
    heartbeat_loss_threshold_range = (3, 15)
    
    # 个体生成函数
    def create_individual():
        # 模块选择部分（二进制）
        modules_part = [random.randint(0, 1) for _ in range(n_modules)]
        
        params_part = [
            random.uniform(*preventive_check_days_range),
            random.uniform(*frequency_heartbeat_range),
            random.uniform(*heartbeat_loss_threshold_range)
        ]
        
        return modules_part + params_part
    
    # 注册个体和种群生成器
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 注册评估函数
    toolbox.register("evaluate", evaluate_individual)
    
    # 注册选择算子（锦标赛选择）
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # 注册交叉算子（两点交叉）
    toolbox.register("mate", tools.cxTwoPoint)
    
    # 注册变异算子
    def mutUniform(individual, indpb):
        n = len(individual)
        
        # 模块部分：按位翻转
        for i in range(n_modules):
            if random.random() < indpb:
                individual[i] = 1 - individual[i]
        
        # 参数部分：均匀变异
        param_indices = range(n_modules, n)
        for i in param_indices:
            if random.random() < indpb:
                if i == n_modules:  # preventive_check_days
                    individual[i] = random.randint(*preventive_check_days_range)
                elif i == n_modules + 1:  # frequency_heartbeat
                    individual[i] = random.uniform(*frequency_heartbeat_range)
                elif i == n_modules + 2:  # heartbeat_loss_threshold
                    individual[i] = random.randint(*heartbeat_loss_threshold_range)
        return individual,
    
    toolbox.register("mutate", mutUniform, indpb=0.1)
    
    # 设置种群规模和迭代次数
    pop_size = 100
    ngen = 100
    
    # 创建初始种群
    pop = toolbox.population(n=pop_size)
    
    # 设置多进程评估
    # if num_processes is None:
    #     num_processes = multiprocessing.cpu_count()
        
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)
    toolbox.register("map", pool.map)
    
    # 评估初始种群
    print(f"Evaluating initial population (using {num_processes} processes)...")
    # 获取评估结果（适应度值和成本明细）
    results = list(toolbox.map(toolbox.evaluate, pop))
    for ind, res in zip(pop, results):
        fitness_value, cost_breakdown = res
        ind.fitness.values = fitness_value
        # 显式存储成本明细到个体
        ind.cost_breakdown = cost_breakdown
    
    # 设置遗传算法参数
    cxpb = 0.7  # 交叉概率
    mutpb = 0.2  # 变异概率
    
    # 创建Hall of Fame记录历史最优解
    hof = tools.HallOfFame(10)
    
    # 记录收敛数据
    convergence_data = {
        'min_total_cost': [],
        'avg_total_cost': [],
        'feasible_count': [],
        'diversity': []
    }
    
    # 记录初始代
    total_costs = [ind.fitness.values[0] for ind in pop]
    feasible_count = sum(1 for cost in total_costs if cost != float('inf'))
    min_cost = min(total_costs)
    
    convergence_data['min_total_cost'].append(min_cost)
    convergence_data['avg_total_cost'].append(
        sum(c for c in total_costs if c != float('inf')) / feasible_count if feasible_count > 0 else float('inf')
    )
    convergence_data['feasible_count'].append(feasible_count)
    
    # 计算初始多样性
    feasible_costs = [c for c in total_costs if c != float('inf')]
    diversity = np.std(feasible_costs) if feasible_costs else 0
    convergence_data['diversity'].append(diversity)
    
    # 运行遗传算法
    print("Starting GA optimization...")
    start_time = time.time()
    
    for gen in range(ngen):
        # 选择下一代个体
        offspring = toolbox.select(pop, len(pop))
        
        # 复制选中的个体
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # 应用交叉操作
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                if hasattr(child1, 'cost_breakdown'):
                    del child1.cost_breakdown
                if hasattr(child2, 'cost_breakdown'):
                    del child2.cost_breakdown
        
        # 应用变异操作
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                if hasattr(mutant, 'cost_breakdown'):
                    del mutant.cost_breakdown
        
        # 评估所有新个体（那些被修改的）
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            results = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, res in zip(invalid_ind, results):
                fitness_value, cost_breakdown = res
                ind.fitness.values = fitness_value
                # 更新成本明细
                ind.cost_breakdown = cost_breakdown
        
        # 替换种群
        pop[:] = offspring
        
        # 更新Hall of Fame
        hof.update(pop)
        
        # 记录当前代数据
        total_costs = [ind.fitness.values[0] for ind in pop]
        feasible_count = sum(1 for cost in total_costs if cost != float('inf'))
        current_min = min(total_costs)
        
        convergence_data['min_total_cost'].append(current_min)
        convergence_data['avg_total_cost'].append(
            sum(c for c in total_costs if c != float('inf')) / feasible_count if feasible_count > 0 else float('inf')
        )
        convergence_data['feasible_count'].append(feasible_count)
        
        # 计算并记录种群多样性
        feasible_costs = [c for c in total_costs if c != float('inf')]
        diversity = np.std(feasible_costs) if feasible_costs else 0
        convergence_data['diversity'].append(diversity)
        
        # 打印进度
        print(f"Gen {gen+1}/{ngen}: Min Cost={current_min:.2f}, "
              f"Feasible={feasible_count}/{pop_size}, "
              f"Diversity={diversity:.1f}")
    
    elapsed = time.time() - start_time
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 从Hall of Fame获取最佳个体
    feasible_individuals = [ind for ind in hof if ind.fitness.values[0] != float('inf')]
    if feasible_individuals:
        best_individual = min(feasible_individuals, key=functools.cmp_to_key(compare_individuals))
    else:
        best_individual = min(pop, key=functools.cmp_to_key(compare_individuals))
    
    return best_individual, pop, elapsed, num_processes, convergence_data

def plot_convergence(convergence_data, elapsed_time, num_procs):
    """绘制收敛图并保存收敛数据"""
    # 确保目录存在
    os.makedirs("results", exist_ok=True)
    
    # 保存收敛数据到JSON文件
    plot_data = {
        "min_total_cost": convergence_data["min_total_cost"],
        "avg_total_cost": convergence_data["avg_total_cost"],
        "feasible_count": convergence_data["feasible_count"],
        "diversity": convergence_data["diversity"],
        "elapsed_time": elapsed_time,
        "num_procs": num_procs
    }
    
    with open("results/ga_convergence_data.json", "w") as f:
        json.dump(plot_data, f, indent=4)
    
    # 1. 总成本收敛图
    plt.figure(figsize=(12, 8))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(convergence_data['min_total_cost'], 'b-', label='Min Total Cost', linewidth=2)
    plt.plot(convergence_data['avg_total_cost'], 'r--', label='Avg Total Cost (Feasible)', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Total Cost', fontsize=16)
    plt.title(f'GA Optimization Convergence\n(Time: {elapsed_time:.2f}s, Processes: {num_procs})', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results/ga_convergence_total_cost.png', dpi=300)
    print("Saved: results/ga_convergence_total_cost.png")
    plt.close()

    # 2. 可行解数量变化
    plt.figure(figsize=(12, 8))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(convergence_data['feasible_count'], 'g-', label='Feasible Solutions', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Feasible Solutions', fontsize=16)
    plt.title('Feasible Solutions Evolution', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results/ga_feasible_count.png', dpi=300)
    print("Saved: results/ga_feasible_count.png")
    plt.close()

    # 3. 多样性变化
    plt.figure(figsize=(12, 8))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})
    
    plt.plot(convergence_data['diversity'], 'c-', label='Diversity (Std Dev)', linewidth=2)
    plt.xlabel('Generation', fontsize=16)
    plt.ylabel('Diversity', fontsize=16)
    plt.title('Fitness Diversity Over Generations', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('results/ga_diversity.png', dpi=300)
    print("Saved: results/ga_diversity.png")
    plt.close()

def replot_from_saved_data():
    """从保存的数据重新绘制图表"""
    try:
        with open("results/ga_convergence_data.json", "r") as f:
            plot_data = json.load(f)
        
        convergence_data = {
            "min_total_cost": plot_data["min_total_cost"],
            "avg_total_cost": plot_data["avg_total_cost"],
            "feasible_count": plot_data["feasible_count"],
            "diversity": plot_data["diversity"]
        }
        
        plot_convergence(convergence_data, plot_data["elapsed_time"], plot_data["num_procs"])
        print("Successfully replotted from saved data")
    except FileNotFoundError:
        print("No saved data found. Please run the optimization first.")
    except Exception as e:
        print(f"Error replotting: {e}")

def extract_solution_info(individual):
    """从个体中提取解决方案信息"""
    n_modules = len(all_modules)
    
    # 提取模块选择
    selected_modules = [all_modules[i] for i in range(n_modules) if individual[i] == 1]
    
    # 提取仿真参数
    param_start_idx = n_modules
    preventive_check_days = int(round(individual[param_start_idx]))
    frequency_heartbeat = int(round(individual[param_start_idx + 1])) if "heartbeat" in selected_modules else None
    heartbeat_loss_threshold = int(round(individual[param_start_idx + 2])) if "heartbeat" in selected_modules else None
    
    # 使用显式存储的成本明细
    cost_info = individual.cost_breakdown
    
    return {
        "modules": selected_modules,
        "preventive_check_days": preventive_check_days,
        "frequency_heartbeat": frequency_heartbeat,
        "heartbeat_loss_threshold": heartbeat_loss_threshold,
        "total_cost": cost_info["total_cost"],
        "base_cost": cost_info["base_cost"],
        "module_cost": cost_info["module_cost"],
        "check_cost": cost_info["check_cost"],
        "fault_cost": cost_info["fault_cost"],
        "data_loss_cost": cost_info["data_loss_cost"],
        "budget": config["budget"]
    }

if __name__ == "__main__":
    # 检查是否只需要重新绘图
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        replot_from_saved_data()
        sys.exit(0)
    
    # 运行遗传算法优化
    print("Starting multi-process GA optimization...")
    
    try:
        best_individual, population, elapsed_time, num_procs, convergence_data = genetic_algorithm_optimization(num_processes=100)
        
        # 绘制收敛图并保存数据
        plot_convergence(convergence_data, elapsed_time, num_procs)
        
        # 提取最佳解决方案
        best_solution = extract_solution_info(best_individual)
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 写入结果文件
        with open("results/ga_optimization_results.txt", "w") as f:
            f.write("Genetic Algorithm Optimization Results (Budget-Constrained)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Optimization Time: {elapsed_time:.2f} seconds | Processes Used: {num_procs}\n")
            f.write(f"Budget: {best_solution['budget']}\n")
            f.write(f"Total Cost: {best_solution['total_cost']:.2f}\n\n")
            
            # 最佳解决方案
            f.write("Best Solution:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Modules: {best_solution['modules']}\n")
            f.write(f"Preventive Check Days: {best_solution['preventive_check_days']} days\n")
            if best_solution['frequency_heartbeat'] is not None:
                f.write(f"Heartbeat Frequency: {best_solution['frequency_heartbeat']} seconds\n")
                f.write(f"Heartbeat Loss Threshold: {best_solution['heartbeat_loss_threshold']}\n")
            f.write("\nCost Breakdown:\n")
            f.write(f"  Base Cost:      {best_solution['base_cost']:.2f}\n")
            f.write(f"  Module Cost:    {best_solution['module_cost']:.2f}\n")
            f.write(f"  Check Cost:     {best_solution['check_cost']:.2f}\n")
            f.write(f"  Fault Cost:     {best_solution['fault_cost']:.2f}\n")
            f.write(f"  Data Loss Cost: {best_solution['data_loss_cost']:.2f}\n")
            f.write(f"  Total Cost:     {best_solution['total_cost']:.2f}\n")
            f.write("-" * 80 + "\n\n")
            
            # 其他可行解（最多前10个）
            feasible_solutions = []
            for ind in population:
                if ind.fitness.values[0] != float('inf'):
                    solution = extract_solution_info(ind)
                    feasible_solutions.append(solution)
            
            feasible_solutions.sort(key=solution_sort_key)
            
            f.write(f"Other Feasible Solutions ({len(feasible_solutions)} total):\n")
            f.write("=" * 80 + "\n")
            for i, sol in enumerate(feasible_solutions[:10]):
                f.write(f"Solution {i+1} (Cost: {sol['total_cost']:.2f}):\n")
                f.write(f"  Modules: {sol['modules']}\n")
                f.write(f"  Preventive Check Days: {sol['preventive_check_days']} days\n")
                if sol['frequency_heartbeat'] is not None:
                    f.write(f"  Heartbeat Frequency: {sol['frequency_heartbeat']} seconds\n")
                    f.write(f"  Heartbeat Loss Threshold: {sol['heartbeat_loss_threshold']}\n")
                f.write(f"  Base Cost:      {sol['base_cost']:.2f}\n")
                f.write(f"  Module Cost:    {sol['module_cost']:.2f}\n")
                f.write(f"  Check Cost:     {sol['check_cost']:.2f}\n") 
                f.write(f"  Fault Cost:     {sol['fault_cost']:.2f}\n")
                f.write(f"  Data Loss Cost: {sol['data_loss_cost']:.2f}\n")
                f.write(f"  Total Cost:     {sol['total_cost']:.2f}\n")
                f.write("-" * 80 + "\n")

        
        # 打印最佳解决方案
        print("\nBest Solution Found:")
        print(f"  Modules: {best_solution['modules']}")
        print(f"  Preventive Check Days: {best_solution['preventive_check_days']} days")
        if best_solution['frequency_heartbeat'] is not None:
            print(f"  Heartbeat Frequency: {best_solution['frequency_heartbeat']} seconds")
            print(f"  Heartbeat Loss Threshold: {best_solution['heartbeat_loss_threshold']}")
        print("  Cost Breakdown:")
        print(f"    Base Cost:      {best_solution['base_cost']:.2f}")
        print(f"    Module Cost:    {best_solution['module_cost']:.2f}")
        print(f"    Check Cost:    {best_solution['check_cost']:.2f}")
        print(f"    Fault Cost:     {best_solution['fault_cost']:.2f}")
        print(f"    Data Loss Cost: {best_solution['data_loss_cost']:.2f}")
        print(f"    Total Cost:     {best_solution['total_cost']:.2f}")
        print(f"  Budget: {best_solution['budget']}")
                
        # 打印执行时间
        print(f"\nGA optimization completed! Time: {elapsed_time:.2f} seconds | Processes: {num_procs}")
        print(f"Feasible solutions found: {len([ind for ind in population if ind.fitness.values[0] != float('inf')])}")
        print("Results saved to 'results' directory")
    
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)