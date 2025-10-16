from utilities import MODULES

def calculate_hardware_cost(nodeNum, hardCost):
    """计算年度硬件成本 （一次性的） - 考虑节点能力和中继节点"""
    # 确保节点数量是整数
    nodeNum = int(round(nodeNum))
        
    return nodeNum*hardCost

#开发成本
def calculate_development_cost(peopleNum,monthly_salary,development_cycle):
    return peopleNum*monthly_salary*development_cycle

#部署成本 installation
def calculate_installation_cost(nodeNum,installation_per_cost):
    return nodeNum*installation_per_cost

def rts_cts_cost(zone_name,config):
    """rts_cts 模块的成本计算逻辑"""
    rts_cts_cycle = MODULES[zone_name]["rts_cts"]["cycle"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,rts_cts_cycle)
    return developmentCost

def heartbeat_cost(zone_name,config):
    """heartbeat 模块的成本计算逻辑"""
    heartbeat_cycle = MODULES[zone_name]["heartbeat"]["cycle"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,heartbeat_cycle)
    return developmentCost

def remote_restart_cost(zone_name,config):
    """remote_restart 模块的成本计算逻辑"""
    remote_restart_cycle = MODULES[zone_name]["remote_restart"]["cycle"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,remote_restart_cycle)
    return developmentCost

def remote_reset_cost(zone_name,config):
    """remote_reset 模块的成本计算逻辑"""
    remote_reset_cycle = MODULES[zone_name]["remote_reset"]["cycle"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,remote_reset_cycle)
    return developmentCost

def boot_update_cost(zone_name,config):
    """boot_update 模块的成本计算逻辑"""
    boot_update_cycle = MODULES[zone_name]["boot_update"]["cycle"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,boot_update_cycle)
    return developmentCost

def noise_cost(zone_name,config):
    """noise 模块的成本计算逻辑"""
    noise_cycle = MODULES[zone_name]["noise"]["cycle"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,noise_cycle)
    return developmentCost

def short_restart_cost(zone_name,config):
    """short_restart 模块的成本计算逻辑"""
    short_restart_cycle = MODULES[zone_name]["short_restart"]["cycle"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,short_restart_cycle)
    return developmentCost

def short_reset_cost(zone_name,config):
    """short_reset 模块的成本计算逻辑"""
    short_reset_cycle = MODULES[zone_name]["short_reset"]["cycle"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,short_reset_cycle)
    return developmentCost

def wireless_power_cost(zone_name,config,zone_configs):
    """wireless_power 模块的成本计算逻辑"""
    wireless_power_cycle = MODULES[zone_name]["wireless_power"]["cycle"]
    sensor_num = zone_configs[zone_name]["sensor_num"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    per_power_cost=zone_configs[zone_name]["per_power_cost"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,wireless_power_cycle)
    hardCost=calculate_hardware_cost(sensor_num,per_power_cost)
    return developmentCost+hardCost

def activation_cost(zone_name,config,zone_configs):
    """activation 模块的成本计算逻辑"""
    activation_cycle = MODULES[zone_name]["activation"]["cycle"]
    sensor_num = zone_configs[zone_name]["sensor_num"]
    developmentPeople = config["development_people"]
    monthly_salary = config["monthly_salary"]
    per_activation_cost=zone_configs[zone_name]["per_activation_cost"]
    developmentCost=calculate_development_cost(developmentPeople,monthly_salary,activation_cycle)
    hardCost=calculate_hardware_cost(sensor_num,per_activation_cost)
    return developmentCost+hardCost

def hardware_wai_cost(zone_config):
    """hardware_wai 模块的成本计算逻辑"""
    sensor_num = zone_config["sensor_num"]
    per_wai_cost=zone_config["per_wai_cost"]
    hardCost=calculate_hardware_cost(sensor_num,per_wai_cost)
    return hardCost

#这个函数放在main那里就可以只执行一次，然后就只是对应的固定模块成本了,计算到utilities的MODULES的cost中，可以根据选择的模块后续进行计算启用的模块的代价
def set_modules_cost(scene_data):
    config = scene_data["config"]
    zone_configs = scene_data["zone_configs"]
    for zone_name, zone_config in zone_configs.items():
        MODULES[zone_name]["rts_cts"]["cost"]=rts_cts_cost(zone_name,config)
        MODULES[zone_name]["heartbeat"]["cost"]=heartbeat_cost(zone_name,config)
        MODULES[zone_name]["remote_restart"]["cost"]=remote_restart_cost(zone_name,config)
        MODULES[zone_name]["remote_reset"]["cost"]=remote_reset_cost(zone_name,config)
        MODULES[zone_name]["boot_update"]["cost"]=boot_update_cost(zone_name,config)
        MODULES[zone_name]["noise"]["cost"]=noise_cost(zone_name,config)
        MODULES[zone_name]["short_restart"]["cost"]=short_restart_cost(zone_name,config)
        MODULES[zone_name]["short_reset"]["cost"]=short_reset_cost(zone_name,config)
        MODULES[zone_name]["wireless_power"]["cost"]=wireless_power_cost(zone_name,config,zone_configs)
        MODULES[zone_name]["activation"]["cost"]=activation_cost(zone_name,config,zone_configs)
        MODULES[zone_name]["hardware_wai"]["cost"]=hardware_wai_cost(zone_config)
     

