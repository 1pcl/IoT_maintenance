from simulator import WSNSimulation    
from calculate_cost import calculate_total_cost_with_simulation
from scene_generator import SceneGenerator

class MaintainabilityCalculator:
    def __init__(self, alpha, beta, total_modules, enabled_modules_by_zone, parameters_by_zone, cost_failure, cost_total):
        """
        初始化可维护性计算器 - 多区域版本
        
        参数:
        alpha (float): 可靠性在可维护性中的权重因子 (0~1)
        beta (float): 模块启用与参数优化的权重因子 (0~1)
        total_modules (int): 总维护模块数量 N_m
        enabled_modules_by_zone (dict): 按区域划分的启用模块字典 {zone: [module1, module2, ...]}
        parameters_by_zone (dict): 按区域划分的参数字典 {zone: [param1, param2, ...]}
        cost_failure (float): 故障总成本 C_failure
        cost_total (float): 系统总成本 C_total
        """
        self.alpha = alpha
        self.beta = beta
        self.total_modules = total_modules
        self.enabled_modules_by_zone = enabled_modules_by_zone
        self.parameters_by_zone = parameters_by_zone
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
    
    def calculate_zone_reliability_score(self, zone):
        """
        计算单个区域的可靠性评分 R_zone (0~1)
        
        参数:
        zone (str): 区域名称
        
        返回:
        float: 区域可靠性评分 R_zone
        """
        # 获取该区域的启用模块和参数
        enabled_modules = self.enabled_modules_by_zone.get(zone, [])
        parameters = self.parameters_by_zone.get(zone, [])
        
        # 计算模块启用比例
        enabled_count = len(enabled_modules)
        module_ratio = enabled_count / self.total_modules
        
        # 计算参数部分
        weighted_sum = 0
        total_weights = 0
        
        for param in parameters:
            # 计入所有权重总和
            total_weights += param['weight']
            
            # 检查参数是否应计入 (δ_k)
            if param['module'] == 'global' or param['module'] in enabled_modules:
                # 计算参数评分并加权
                param_score = self.calculate_parameter_score(param)
                weighted_sum += param['weight'] * param_score
        
        # 避免除以零
        if total_weights == 0:
            normalized_param_score = 0
        else:
            normalized_param_score = weighted_sum / total_weights
        
        # 组合两部分评分
        R_zone = self.beta * module_ratio + (1 - self.beta) * normalized_param_score
        
        # 确保在[0,1]范围内
        return max(0, min(R_zone, 1))
    
    def calculate_system_reliability_score(self):
        """
        计算系统整体可靠性评分 R_system (各区域可靠性的平均值)
        
        返回:
        float: 系统可靠性评分 R_system
        """
        zone_reliabilities = []
        
        for zone in self.enabled_modules_by_zone.keys():
            zone_reliability = self.calculate_zone_reliability_score(zone)
            zone_reliabilities.append(zone_reliability)
        
        # 计算各区域可靠性的平均值
        if zone_reliabilities:
            R_system = sum(zone_reliabilities) / len(zone_reliabilities)
        else:
            R_system = 0
        
        return R_system
    
    def calculate_maintainability(self):
        """
        计算系统可维护性评分 M (0~1)
        
        返回:
        float: 可维护性评分 M
        """
        # 计算系统可靠性评分 R_system
        R_system = self.calculate_system_reliability_score()
        
        # 计算成本比率部分
        cost_ratio = 1 - (self.cost_failure / self.cost_total)
        
        # 计算最终可维护性评分
        M = self.alpha * R_system + (1 - self.alpha) * cost_ratio
        
        # 确保结果在合理范围内
        return max(0, min(M, 1))
    
    def get_zone_reliability_scores(self):
        """
        获取各区域的可靠性评分
        
        返回:
        dict: {zone: reliability_score}
        """
        zone_scores = {}
        for zone in self.enabled_modules_by_zone.keys():
            zone_scores[zone] = self.calculate_zone_reliability_score(zone)
        return zone_scores

def run_single_simulation():
    """运行单次仿真并返回结果"""
    scene_data = SceneGenerator.create_scene(visualize=False)
    
    # 按区域配置维护模块
    selected_modules_by_zone = {
        "zone_1": ["heartbeat", "rts_cts", "wireless_power", "remote_restart"],
        "zone_2": ["heartbeat", "remote_restart"],
        "zone_3": ["heartbeat", "rts_cts", "short_restart"],
        "zone_4": ["heartbeat"]
    }
    
    # 按区域配置仿真参数
    simulation_params_by_zone = {
        "zone_1": {
            "preventive_check_days": 1,
            "frequency_heartbeat": 1,  # 每分钟1次
            "heartbeat_loss_threshold": 3
        },
        "zone_2": {
            "preventive_check_days": 3,
            "frequency_heartbeat": 1/5,  # 每5分钟1次
            "heartbeat_loss_threshold": 3
        },
        "zone_3": {
            "preventive_check_days": 2,
            "frequency_heartbeat": 1/15,  # 每15分钟1次
            "heartbeat_loss_threshold": 3
        },
        "zone_4": {
            "preventive_check_days": 7,
            "frequency_heartbeat": 1/30,  # 每30分钟1次
            "heartbeat_loss_threshold": 3
        }
    }

    # 创建仿真实例
    sim = WSNSimulation(scene_data, selected_modules_by_zone, simulation_params_by_zone)

    # 计算总成本
    total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost = calculate_total_cost_with_simulation(
        sim, selected_modules_by_zone, scene_data
    )
    
    return total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost, selected_modules_by_zone, simulation_params_by_zone

def calculate_maintainability_for_run(cost_failure, cost_total, selected_modules_by_zone, simulation_params_by_zone):
    """计算单次运行的可维护性得分 - 多区域版本"""
    # 定义系统参数
    alpha = 0.3
    beta = 0.6
    total_modules = 11  # 每个区域的总模块数

    # 为每个区域构建参数列表
    parameters_by_zone = {}
    
    for zone, params in simulation_params_by_zone.items():
        zone_parameters = [
            # 区域特定的预防性检查天数
            {
                'name': f'{zone}_preventive_check_days', 
                'value': params["preventive_check_days"], 
                'min': 1, 
                'max': 180, 
                'weight': 0.7, 
                'flag': 0,  # 负相关 - 天数越少越好
                'module': 'global'
            },
        ]
        
        # 只有在包含heartbeat模块时才添加相关参数
        if 'heartbeat' in selected_modules_by_zone.get(zone, []):
            zone_parameters.extend([
                {
                    'name': f'{zone}_frequency_heartbeat', 
                    'value': params["frequency_heartbeat"], 
                    'min': 1/60,  # 1秒
                    'max': 60,    # 60秒
                    'weight': 0.1, 
                    'flag': 0,    # 负相关 - 频率越低越好（节能）
                    'module': 'heartbeat'
                },
                {
                    'name': f'{zone}_heartbeat_loss_threshold', 
                    'value': params["heartbeat_loss_threshold"], 
                    'min': 3, 
                    'max': 15, 
                    'weight': 0.2, 
                    'flag': 0,    # 负相关 - 阈值越低越敏感
                    'module': 'heartbeat'
                },
            ])
        
        parameters_by_zone[zone] = zone_parameters
    
    # 创建计算器实例
    calculator = MaintainabilityCalculator(
        alpha=alpha,
        beta=beta,
        total_modules=total_modules,
        enabled_modules_by_zone=selected_modules_by_zone,
        parameters_by_zone=parameters_by_zone,
        cost_failure=cost_failure,
        cost_total=cost_total
    )
    
    # 计算可维护性评分和系统可靠性评分
    maintainability_score = calculator.calculate_maintainability()
    system_reliability_score = calculator.calculate_system_reliability_score()
    
    # 获取各区域的可靠性评分
    zone_reliability_scores = calculator.get_zone_reliability_scores()
    
    return maintainability_score, system_reliability_score, zone_reliability_scores

def main():
    # 运行50次仿真
    num_runs = 50
    results = []
    
    print(f"开始运行 {num_runs} 次仿真...")
    
    for i in range(num_runs):
        print(f"正在运行第 {i+1} 次仿真...")
        try:
            total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost, selected_modules_by_zone, simulation_params_by_zone = run_single_simulation()
            
            # 计算可维护性得分（多区域版本）
            maintainability_score, system_reliability_score, zone_reliability_scores = calculate_maintainability_for_run(
                fault_cost, total_cost, selected_modules_by_zone, simulation_params_by_zone
            )
            
            result = {
                'run_id': i + 1,
                'total_cost': total_cost,
                'base_cost': base_cost,
                'module_cost': module_cost,
                'check_cost': check_cost,
                'fault_cost': fault_cost,
                'data_loss_cost': data_loss_cost,
                'selected_modules_by_zone': selected_modules_by_zone.copy(),
                'simulation_params_by_zone': simulation_params_by_zone.copy(),
                'maintainability_score': maintainability_score,
                'system_reliability_score': system_reliability_score,
                'zone_reliability_scores': zone_reliability_scores.copy()
            }
            results.append(result)
            
            print(f"第 {i+1} 次运行完成 - 总成本: {total_cost:.2f}, 可维护性得分: {maintainability_score:.4f}, 系统可靠性: {system_reliability_score:.4f}")
            
        except Exception as e:
            print(f"第 {i+1} 次运行失败: {e}")
            continue
    
    # 写入结果到txt文件
    with open("simulation_results_multi_zone.txt", "w", encoding="utf-8") as f:
        f.write("多区域仿真结果统计\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'运行次数':<8} {'总成本':<12} {'故障成本':<12} {'启用模块总数':<15} {'可维护性得分':<15} {'系统可靠性':<15}")
        
        # 添加各区域可靠性列
        zones = ["zone_1", "zone_2", "zone_3", "zone_4"]
        for zone in zones:
            f.write(f" {zone+'_可靠性':<15}")
        f.write("\n")
        
        f.write("-" * 150 + "\n")
        
        for result in results:
            f.write(f"{result['run_id']:<10} {result['total_cost']:<12.2f} {result['fault_cost']:<12.2f} "
                   f"{sum(len(modules) for modules in result['selected_modules_by_zone'].values()):<15} "
                   f"{result['maintainability_score']:<15.4f} {result['system_reliability_score']:<15.4f}")
            
            # 写入各区域可靠性
            for zone in zones:
                reliability = result['zone_reliability_scores'].get(zone, 0)
                f.write(f" {reliability:<15.4f}")
            f.write("\n")
        
        # 计算平均值
        if results:
            avg_total_cost = sum(r['total_cost'] for r in results) / len(results)
            avg_base_cost = sum(r['base_cost'] for r in results) / len(results)
            avg_module_cost = sum(r['module_cost'] for r in results) / len(results)
            avg_check_cost = sum(r['check_cost'] for r in results) / len(results)
            avg_fault_cost = sum(r['fault_cost'] for r in results) / len(results)
            avg_data_loss_cost = sum(r['data_loss_cost'] for r in results) / len(results)
            avg_module_count = sum(sum(len(modules) for modules in r['selected_modules_by_zone'].values()) for r in results) / len(results)
            avg_maintainability = sum(r['maintainability_score'] for r in results) / len(results)
            avg_system_reliability = sum(r['system_reliability_score'] for r in results) / len(results)
            
            # 计算各区域平均可靠性
            avg_zone_reliabilities = {}
            for zone in zones:
                zone_reliabilities = [r['zone_reliability_scores'].get(zone, 0) for r in results]
                avg_zone_reliabilities[zone] = sum(zone_reliabilities) / len(zone_reliabilities)
            
            f.write("-" * 150 + "\n")
            f.write("平均值统计:\n")
            f.write(f"平均总成本: {avg_total_cost:.2f}\n")
            f.write(f"平均基础成本: {avg_base_cost:.2f}\n")
            f.write(f"平均模块成本: {avg_module_cost:.2f}\n")
            f.write(f"平均检查成本: {avg_check_cost:.2f}\n")
            f.write(f"平均故障成本: {avg_fault_cost:.2f}\n")
            f.write(f"平均数据丢失成本: {avg_data_loss_cost:.2f}\n")
            f.write(f"平均启用模块总数: {avg_module_count:.1f}\n")
            f.write(f"平均可维护性得分: {avg_maintainability:.4f}\n")
            f.write(f"平均系统可靠性: {avg_system_reliability:.4f}\n")
            
            # 写入各区域平均可靠性
            f.write("各区域平均可靠性:\n")
            for zone, reliability in avg_zone_reliabilities.items():
                f.write(f"  {zone}: {reliability:.4f}\n")
            
            # 在控制台也输出平均值
            print("\n" + "="*70)
            print("平均值统计:")
            print(f"平均总成本: {avg_total_cost:.2f}")
            print(f"平均基础成本: {avg_base_cost:.2f}")
            print(f"平均模块成本: {avg_module_cost:.2f}")
            print(f"平均检查成本: {avg_check_cost:.2f}")
            print(f"平均故障成本: {avg_fault_cost:.2f}")
            print(f"平均数据丢失成本: {avg_data_loss_cost:.2f}")
            print(f"平均启用模块总数: {avg_module_count:.1f}")
            print(f"平均可维护性得分: {avg_maintainability:.4f}")
            print(f"平均系统可靠性: {avg_system_reliability:.4f}")
            print("各区域平均可靠性:")
            for zone, reliability in avg_zone_reliabilities.items():
                print(f"  {zone}: {reliability:.4f}")
            print(f"结果已保存到 simulation_results_multi_zone.txt")
    
    return results

if __name__ == "__main__":
    results = main()