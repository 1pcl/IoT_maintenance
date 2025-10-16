from utilities import FAULT_TYPES, MODULES, DEFAULT_PARAM_VALUES
from scene_generator import SceneGenerator
from simulator import WSNSimulation    
from calculate_cost import calculate_total_cost_with_simulation

# 使用示例
if __name__ == "__main__":
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
            "preventive_check_days": 30,
            "frequency_heartbeat": 10,  
            "heartbeat_loss_threshold": 3
        },
        "zone_2": {
            "preventive_check_days": 30,
            "frequency_heartbeat": 50,  
            "heartbeat_loss_threshold": 3
        },
        "zone_3": {
            "preventive_check_days": 30,
            "frequency_heartbeat": 500,  
            "heartbeat_loss_threshold": 3
        },
        "zone_4": {
            "preventive_check_days": 30,
            "frequency_heartbeat": 1000,  
            "heartbeat_loss_threshold": 3
        }
    }

    # 创建仿真实例
    sim = WSNSimulation(scene_data, selected_modules_by_zone, simulation_params_by_zone)

    # loss_data_count, node_fault_list, check_count = sim.run_simulation()
    # 计算总成本
    total_cost, base_cost, module_cost, check_cost, fault_cost, data_loss_cost = calculate_total_cost_with_simulation(
        sim, selected_modules_by_zone, scene_data
    )

    print(f"  总成本: {total_cost:.2f}")
    print(f"  基本成本: {base_cost:.2f}")
    print(f"  模块成本: {module_cost:.2f}")
    print(f"  检查成本: {check_cost:.2f}")
    print(f"  故障成本: {fault_cost:.2f}")
    print(f"  数据丢失成本: {data_loss_cost:.2f}")