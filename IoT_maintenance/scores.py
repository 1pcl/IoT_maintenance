from simulator import WSNSimulation    
from calculate_cost import calculate_total_cost_with_simulation
from module_cost import set_modules_cost
from scene_generator import SceneGenerator

class MaintainabilityCalculator:
    def __init__(self, alpha, beta, total_modules, enabled_modules, parameters, cost_failure, cost_total):
        """
        初始化可维护性计算器
        
        参数:
        alpha (float): 可靠性在可维护性中的权重因子 (0~1)
        beta (float): 模块启用与参数优化的权重因子 (0~1)
        total_modules (int): 总维护模块数量 N_m
        enabled_modules (list): 启用的模块名称列表
        parameters (list of dict): 关键参数列表，每个字典包含:
            'name': 参数名
            'value': 当前值
            'min': 允许的最小值
            'max': 允许的最大值
            'weight': 权重 w_k
            'flag': 相关性标志 (1=正相关, 0=负相关)
            'module': 所属模块名 ('global' 表示全局参数)
        cost_failure (float): 故障总成本 C_failure
        cost_total (float): 系统总成本 C_total
        """
        self.alpha = alpha
        self.beta = beta
        self.total_modules = total_modules
        self.enabled_modules = enabled_modules
        self.parameters = parameters
        self.cost_failure = cost_failure
        self.cost_total = cost_total
        
        # 验证输入有效性
        if not (0 <= alpha <= 1):
            raise ValueError("alpha 必须在 [0, 1] 范围内")
        if not (0 <= beta <= 1):
            raise ValueError("beta 必须在 [0, 1] 范围内")
        if total_modules <= 0:
            raise ValueError("总模块数必须大于0")
        if cost_total <= 0:
            raise ValueError("总成本必须大于0")
        if cost_failure < 0:
            raise ValueError("故障成本不能为负数")
        if cost_failure > cost_total:
            raise ValueError("故障成本不能超过总成本")
    
    def calculate_parameter_score(self, param):
        """
        计算单个参数的可靠性评分 s_i
        
        参数:
        param (dict): 参数属性字典
        
        返回:
        float: 归一化的参数评分 (0~1)
        """
        value = param['value']
        min_val = param['min']
        max_val = param['max']
        
        # 处理无效值范围
        if max_val <= min_val:
            raise ValueError(f"参数 {param['name']} 的最大值必须大于最小值")
        
        # 确保参数值在有效范围内
        value = max(min_val, min(value, max_val))
        
        # 根据相关性标志计算评分
        if param['flag'] == 1:  # 正相关
            return (value - min_val) / (max_val - min_val)
        else:  # 负相关 (flag=0)
            return (max_val - value) / (max_val - min_val)
    
    def calculate_reliability_score(self):
        """
        计算系统可靠性评分 R (0~1)
        
        返回:
        float: 可靠性评分 R
        """
        # 计算模块启用比例
        enabled_count = len(self.enabled_modules)
        module_ratio = enabled_count / self.total_modules
        
        # 计算参数部分
        weighted_sum = 0
        total_weights = 0
        
        for param in self.parameters:
            # 计入所有权重总和
            total_weights += param['weight']
            
            # 检查参数是否应计入 (δ_k)
            if param['module'] == 'global' or param['module'] in self.enabled_modules:
                # 计算参数评分并加权
                param_score = self.calculate_parameter_score(param)
                weighted_sum += param['weight'] * param_score
        
        # 避免除以零
        if total_weights == 0:
            normalized_param_score = 0
        else:
            normalized_param_score = weighted_sum / total_weights
        
        # 组合两部分评分
        R = self.beta * module_ratio + (1 - self.beta) * normalized_param_score
        
        # 确保在[0,1]范围内
        return max(0, min(R, 1))
    
    def calculate_maintainability(self):
        """
        计算系统可维护性评分 M (0~1)
        
        返回:
        float: 可维护性评分 M
        """
        # 计算可靠性评分 R
        R = self.calculate_reliability_score()
        
        # 计算成本比率部分
        cost_ratio = 1 - (self.cost_failure / self.cost_total)
        
        # 计算最终可维护性评分
        M = self.alpha * R + (1 - self.alpha) * cost_ratio
        
        # 确保结果在合理范围内
        return max(0, min(M, 1))

def run_single_simulation():
    """运行单次仿真并返回结果"""
    application = "electricity_meter"  # animal_room, electricity_meter
    scene_data = SceneGenerator.create_scene(application, visualize=False)
    config = scene_data["config"]
    
    # 统一的模块列表
    selected_modules = ['rts_cts', 'boot_update', 'hardware_wai', 'remote_restart', 'remote_reset', 'short_restart', 'short_reset']
    
    # 设置模块成本
    set_modules_cost(config)

    simulation_params = {
        "preventive_check_days": 170,
        # "frequency_heartbeat": 60,
        # "heartbeat_loss_threshold": 5
    }

    # 创建仿真实例
    sim = WSNSimulation(scene_data, selected_modules, simulation_params)

    # 计算总成本
    total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost = calculate_total_cost_with_simulation(
        sim, selected_modules, config
    )
    
    return total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost, selected_modules

def calculate_maintainability_for_run(cost_failure, cost_total, selected_modules):
    """计算单次运行的可维护性得分"""
    # 定义系统参数
    alpha = 0.3
    beta = 0.6
    total_modules = 11

    # 使用统一的模块列表
    enabled_modules = selected_modules
    
    # 定义基础参数列表（全局参数）
    parameters = [
        # 全局参数
        {'name': 'preventive_check_days', 'value': 170, 'min': 1, 'max': 180, 'weight': 0.7, 'flag': 0, 'module': 'global'},
    ]
    
    # 只有在包含heartbeat模块时才添加相关参数
    if 'heartbeat' in selected_modules:
        parameters.extend([
            {'name': 'frequency_heartbeat', 'value': 30, 'min': 1, 'max': 50, 'weight': 0.1, 'flag': 0, 'module': 'heartbeat'},
            {'name': 'heartbeat_loss_threshold', 'value': 75, 'min': 3, 'max': 15, 'weight': 0.2, 'flag': 0, 'module': 'heartbeat'},
        ])
    
    # 创建计算器实例
    calculator = MaintainabilityCalculator(
        alpha=alpha,
        beta=beta,
        total_modules=total_modules,
        enabled_modules=enabled_modules,
        parameters=parameters,
        cost_failure=cost_failure,
        cost_total=cost_total
    )
    
    # 计算可维护性评分
    maintainability_score = calculator.calculate_maintainability()
    reliability_score = calculator.calculate_reliability_score()
    
    return maintainability_score, reliability_score

def main():
    # 运行50次仿真
    num_runs = 50
    results = []
    
    print(f"开始运行 {num_runs} 次仿真...")
    
    for i in range(num_runs):
        print(f"正在运行第 {i+1} 次仿真...")
        try:
            total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost, selected_modules = run_single_simulation()
            
            # 计算可维护性得分
            maintainability_score, reliability_score = calculate_maintainability_for_run(fault_cost, total_cost, selected_modules)
            
            result = {
                'run_id': i + 1,
                'total_cost': total_cost,
                'base_cost': base_cost,
                'module_cost': module_cost,
                'check_cost': check_cost,
                'fault_cost': fault_cost,
                'data_loss_cost': data_loss_cost,
                'selected_modules': selected_modules.copy(),  # 保存模块列表
                'maintainability_score': maintainability_score,
                'reliability_score': reliability_score
            }
            results.append(result)
            
            print(f"第 {i+1} 次运行完成 - 总成本: {total_cost:.2f}, 可维护性得分: {maintainability_score:.4f}")
            
        except Exception as e:
            print(f"第 {i+1} 次运行失败: {e}")
            continue
    
    # 写入结果到txt文件
    with open("simulation_results.txt", "w", encoding="utf-8") as f:
        f.write("仿真结果统计 (30次运行)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'运行次数':<8} {'总成本':<12} {'基础成本':<12} {'模块成本':<12} {'检查成本':<12} {'故障成本':<12} {'数据丢失成本':<15} {'启用模块数':<12} {'可维护性得分':<15} {'可靠性得分':<15}\n")
        f.write("-" * 160 + "\n")
        
        for result in results:
            f.write(f"{result['run_id']:<10} {result['total_cost']:<12.2f} {result['base_cost']:<12.2f} {result['module_cost']:<12.2f} "
                   f"{result['check_cost']:<12.2f} {result['fault_cost']:<12.2f} {result['data_loss_cost']:<15.2f} "
                   f"{len(result['selected_modules']):<12} {result['maintainability_score']:<15.4f} {result['reliability_score']:<15.4f}\n")
        
        # 计算平均值
        if results:
            avg_total_cost = sum(r['total_cost'] for r in results) / len(results)
            avg_base_cost = sum(r['base_cost'] for r in results) / len(results)
            avg_module_cost = sum(r['module_cost'] for r in results) / len(results)
            avg_check_cost = sum(r['check_cost'] for r in results) / len(results)
            avg_fault_cost = sum(r['fault_cost'] for r in results) / len(results)
            avg_data_loss_cost = sum(r['data_loss_cost'] for r in results) / len(results)
            avg_module_count = sum(len(r['selected_modules']) for r in results) / len(results)
            avg_maintainability = sum(r['maintainability_score'] for r in results) / len(results)
            avg_reliability = sum(r['reliability_score'] for r in results) / len(results)
            
            f.write("-" * 160 + "\n")
            f.write("平均值统计:\n")
            f.write(f"平均总成本: {avg_total_cost:.2f}\n")
            f.write(f"平均基础成本: {avg_base_cost:.2f}\n")
            f.write(f"平均模块成本: {avg_module_cost:.2f}\n")
            f.write(f"平均检查成本: {avg_check_cost:.2f}\n")
            f.write(f"平均故障成本: {avg_fault_cost:.2f}\n")
            f.write(f"平均数据丢失成本: {avg_data_loss_cost:.2f}\n")
            f.write(f"平均启用模块数: {avg_module_count:.1f}\n")
            f.write(f"平均可维护性得分: {avg_maintainability:.4f}\n")
            f.write(f"平均可靠性得分: {avg_reliability:.4f}\n")
            
            # 在控制台也输出平均值
            print("\n" + "="*50)
            print("平均值统计:")
            print(f"平均总成本: {avg_total_cost:.2f}")
            print(f"平均基础成本: {avg_base_cost:.2f}")
            print(f"平均模块成本: {avg_module_cost:.2f}")
            print(f"平均检查成本: {avg_check_cost:.2f}")
            print(f"平均故障成本: {avg_fault_cost:.2f}")
            print(f"平均数据丢失成本: {avg_data_loss_cost:.2f}")
            print(f"平均启用模块数: {avg_module_count:.1f}")
            print(f"平均可维护性得分: {avg_maintainability:.4f}")
            print(f"平均可靠性得分: {avg_reliability:.4f}")
            print(f"结果已保存到 simulation_results.txt")
    
    return results

if __name__ == "__main__":
    results = main()