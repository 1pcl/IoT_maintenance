class MaintainabilityCalculator:
    def __init__(self, alpha, beta, total_modules, enabled_modules, parameters, cost_failure, cost_total):
        """
        初始化可维护性计算器
        
        参数:
        alpha (float): 可靠性在可维护性中的权重因子 (0~1)
        beta (float): 模块启用与参数优化的权重因子 (0~1)
        total_modules (int): 总维护模块数量 N_m
        enabled_modules (list): 启用的模块名称列表
        parameters (list of dict): 关键参数列表
        cost_failure (float): 故障总成本 C_failure（包括故障代价和数据丢失代价）
        cost_total (float): 系统总成本 C_total
        """
        self.alpha = alpha
        self.beta = beta
        self.total_modules = total_modules
        self.enabled_modules = enabled_modules
        self.parameters = parameters
        self.cost_failure = cost_failure
        self.cost_total = cost_total
    
    def calculate_parameter_score(self, param):
        """
        计算单个参数的可靠性评分 s_i
        """
        value = param['value']
        min_val = param['min']
        max_val = param['max']
        
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
        """
        # 计算模块启用比例
        enabled_count = len(self.enabled_modules)
        module_ratio = enabled_count / self.total_modules
        
        # 计算参数部分
        weighted_sum = 0
        total_weights = 0
        
        for param in self.parameters:
            # 检查参数是否应计入 (δ_k)
            if param['module'] == 'global' or param['module'] in self.enabled_modules:
                total_weights += param['weight']
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
        """
        # 计算可靠性评分 R
        R = self.calculate_reliability_score()
        
        # 计算成本比率部分
        cost_ratio = 1 - (self.cost_failure / self.cost_total)
        
        # 计算最终可维护性评分
        M = self.alpha * R + (1 - self.alpha) * cost_ratio
        
        # 确保结果在合理范围内
        return max(0, min(M, 1))

def calculate_from_file(file_path):
    """
    从仿真结果文件计算可靠性和可维护性
    
    参数:
    file_path (str): 仿真结果文件路径
    """
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析数据行
    results = []
    for line in lines:
        # 跳过表头和分隔线
        if line.strip() and not line.startswith(('仿真结果', '=', '-', '运行次数', '平均值')):
            parts = line.split()
            if len(parts) >= 7:  # 确保有足够的列
                try:
                    result = {
                        'run_id': int(parts[0]),
                        'total_cost': float(parts[1]),
                        'base_cost': float(parts[2]),
                        'module_cost': float(parts[3]),
                        'check_cost': float(parts[4]),
                        'fault_cost': float(parts[5]),
                        'data_loss_cost': float(parts[6])
                    }
                    results.append(result)
                except (ValueError, IndexError):
                    continue
    
    print(f"成功读取 {len(results)} 条数据")
    
    # 0.3 or 0.4 , 0.6
    alpha = 0.4
    beta = 0.7
    total_modules = 11
    
    # 启用模块列表
    enabled_modules =['rts_cts', 'boot_update', 'hardware_wai', 'remote_restart', 'remote_reset', 'short_restart', 'short_reset']
    
    # 参数配置
    parameters = [
        # 全局参数 'weight': 0.4 0r  0.7   or   1
        {'name': 'preventive_check_days', 'value': 162, 'min': 1, 'max': 180, 'weight': 1, 'flag': 0, 'module': 'global'},
        
        # 模块参数（只有启用对应模块时才会计入）meter 30 1800,animal:10 600
        # {'name': 'frequency_heartbeat', 'value': 150, 'min': 10, 'max': 600, 'weight': 0.1, 'flag': 0, 'module': 'heartbeat'},
        # {'name': 'heartbeat_loss_threshold', 'value': 8, 'min': 3, 'max': 15, 'weight': 0.5, 'flag': 0, 'module': 'heartbeat'},
    ]
    
    # 为每条数据计算可维护性得分
    calculated_results = []
    for result in results:
        # 故障总成本 = 故障代价 + 数据丢失代价
        total_failure_cost = result['fault_cost'] + result['data_loss_cost']
        
        calculator = MaintainabilityCalculator(
            alpha=alpha,
            beta=beta,
            total_modules=total_modules,
            enabled_modules=enabled_modules,
            parameters=parameters,
            cost_failure=total_failure_cost,  
            cost_total=result['total_cost']
        )
        
        maintainability_score = calculator.calculate_maintainability()
        reliability_score = calculator.calculate_reliability_score()
        
        calculated_result = {
            **result,
            'total_failure_cost': total_failure_cost,  # 添加故障总成本字段
            'maintainability_score': maintainability_score,
            'reliability_score': reliability_score
        }
        calculated_results.append(calculated_result)
    
    # 输出结果到新文件
    output_file = "calculated_maintainability_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("基于成本数据的可维护性计算结果\n")
        f.write("=" * 100 + "\n")
        f.write("使用的参数配置:\n")
        f.write(f"  alpha (可靠性权重): {alpha}\n")
        f.write(f"  beta (模块权重): {beta}\n")
        f.write(f"  总模块数: {total_modules}\n")
        f.write(f"  启用模块: {', '.join(enabled_modules)}\n")
        f.write(f"  启用模块数: {len(enabled_modules)}\n")
        f.write(f"  参数数量: {len(parameters)}\n")
        f.write("=" * 100 + "\n")
        
        f.write(f"{'运行次数':<8} {'总成本':<12} {'故障代价':<12} {'数据丢失':<12} {'故障总成本':<12} {'可维护性得分':<15} {'可靠性得分':<15}\n")
        f.write("-" * 90 + "\n")
        
        for result in calculated_results:
            f.write(f"{result['run_id']:<10} {result['total_cost']:<12.2f} {result['fault_cost']:<12.2f} "
                   f"{result['data_loss_cost']:<12.2f} {result['total_failure_cost']:<12.2f} "
                   f"{result['maintainability_score']:<15.4f} {result['reliability_score']:<15.4f}\n")
        
        # 计算平均值
        if calculated_results:
            avg_total_cost = sum(r['total_cost'] for r in calculated_results) / len(calculated_results)
            avg_fault_cost = sum(r['fault_cost'] for r in calculated_results) / len(calculated_results)
            avg_data_loss = sum(r['data_loss_cost'] for r in calculated_results) / len(calculated_results)
            avg_total_failure = sum(r['total_failure_cost'] for r in calculated_results) / len(calculated_results)
            avg_maintainability = sum(r['maintainability_score'] for r in calculated_results) / len(calculated_results)
            avg_reliability = sum(r['reliability_score'] for r in calculated_results) / len(calculated_results)
            
            f.write("-" * 90 + "\n")
            f.write("平均值统计:\n")
            f.write(f"平均总成本: {avg_total_cost:.2f}\n")
            f.write(f"平均故障代价: {avg_fault_cost:.2f}\n")
            f.write(f"平均数据丢失代价: {avg_data_loss:.2f}\n")
            f.write(f"平均故障总成本: {avg_total_failure:.2f}\n")
            f.write(f"平均可维护性得分: {avg_maintainability:.4f}\n")
            f.write(f"平均可靠性得分: {avg_reliability:.4f}\n")
    
    print(f"计算结果已保存到 {output_file}")
    print(f"平均可维护性得分: {avg_maintainability:.4f}")
    print(f"平均可靠性得分: {avg_reliability:.4f}")
    
    return calculated_results

def main():
    # 直接指定文件路径
    file_path = "meter_ga_cost.txt" 
    
    try:
        results = calculate_from_file(file_path)
        
        # 在控制台显示前几条结果
        print("\n前5条计算结果:")
        print(f"{'运行次数':<8} {'总成本':<12} {'故障代价':<12} {'数据丢失':<12} {'故障总成本':<12} {'可维护性得分':<15} {'可靠性得分':<15}")
        print("-" * 90)
        for result in results[:5]:
            print(f"{result['run_id']:<10} {result['total_cost']:<12.2f} {result['fault_cost']:<12.2f} "
                  f"{result['data_loss_cost']:<12.2f} {result['total_failure_cost']:<12.2f} "
                  f"{result['maintainability_score']:<15.4f} {result['reliability_score']:<15.4f}")
                  
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"计算过程中出现错误: {e}")

if __name__ == "__main__":
    main()