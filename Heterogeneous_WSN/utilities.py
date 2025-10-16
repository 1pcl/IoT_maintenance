# -*- coding: utf-8 -*-
#同一个网络有四种类型的传感器
# 定义所有可用维护模块及其属性(根据传感器类型差异化)
MODULES = {
    "zone_1": {
        "rts_cts": {
            "name": "RTS/CTS mechanism",
            "cycle": 0.008,  # 更频繁的RTS/CTS，工业环境干扰大
            "cost": 0,       # 工业级模块成本更高
            "prevents": ["hidden node failure"],
        },
        "heartbeat": {
            "name": "node heartbeat monitoring",
            "cycle": 0.003,  # 高频心跳，实时监控需求
            "cost": 0,
            "prevents": ["power failure"],
        },
        "boot_update": {            
            "name": "program/firmware remote update",
            "cycle": 0.004,  # 频繁固件更新，工业设备需要最新算法
            "cost": 0,
            "prevents": ["boot_fault"],
        },
        "wireless_power": {
            "name": "wireless charging module",
            "cycle": 0.005,  # 工业设备功耗大，需要更频繁充电
            "cost": 0,      # 工业级无线充电成本高
            "prevents": ["power failure"],
        },
        "hardware_wai": {
            "name": "peripheral hardware maintenance",
            "cost": 0,      # 工业硬件维护成本高
            "reduces": {"hardware failure": 0.80},  # 工业环境恶劣，效果略低
        },
        "remote_restart": {
            "name": "remote restart",
            "cycle": 0.004,
            "cost": 0,
            "fixed_success": {"data failure": 0.65, "data loss failure": 0.65}, 
        },
        "remote_reset": {
            "name": "remote restore factory settings",
            "cycle": 0.004,
            "cost": 0,
            "fixed_success": {"data failure": 0.75, "data loss failure": 0.75},  
        },
        "noise": {
            "name": "recalibration of background noise",
            "cycle": 0.08,   # 工业噪声大，需要频繁校准
            "cost": 0,
            "fixed_success": {"data failure": 0.55},  
        },
        "short_restart": {
            "name": "close restart module",
            "cycle": 0.004,
            "cost": 0,
            "fixed_success": {"communication failure": 0.75},  
        },
        "short_reset": {
            "name": "short-range restore factory Settings",
            "cycle": 0.004,
            "cost": 0,
            "fixed_success": {"communication failure": 0.75},  
        },
        "activation": {
            "name": "sensor reactivation",
            "cycle": 0.10,   # 工业设备需要频繁激活
            "cost": 0,      # 工业设备激活成本高
            "fixed_success": {"communication failure": 0.65},  
        },
    },
    "zone_2": {
        "rts_cts": {
            "name": "RTS/CTS mechanism",
            "cycle": 0.012,  # 环境相对稳定，RTS/CTS频率较低
            "cost": 0,
            "prevents": ["hidden node failure"],
        },
        "heartbeat": {
            "name": "node heartbeat monitoring",
            "cycle": 0.006,  # 中等频率心跳
            "cost": 0,
            "prevents": ["power failure"],
        },
        "boot_update": {            
            "name": "program/firmware remote update",
            "cycle": 0.006,  # 固件更新频率中等
            "cost": 0,
            "prevents": ["boot_fault"],
        },
        "wireless_power": {
            "name": "wireless charging module",
            "cycle": 0.007,  # 中等功耗
            "cost": 0,
            "prevents": ["power failure"],
        },
        "hardware_wai": {
            "name": "peripheral hardware maintenance",
            "cost": 0,
            "reduces": {"hardware failure": 0.88},  # 环境较好，维护效果更好
        },
        "remote_restart": {
            "name": "remote restart",
            "cycle": 0.006,
            "cost": 0,
            "fixed_success": {"data failure": 0.72, "data loss failure": 0.72}, 
        },
        "remote_reset": {
            "name": "remote restore factory settings",
            "cycle": 0.006,
            "cost": 0,
            "fixed_success": {"data failure": 0.82, "data loss failure": 0.82},  
        },
        "noise": {
            "name": "recalibration of background noise",
            "cycle": 0.12,   # 环境噪声小，校准频率低
            "cost": 0,
            "fixed_success": {"data failure": 0.65},  
        },
        "short_restart": {
            "name": "close restart module",
            "cycle": 0.006,
            "cost": 0,
            "fixed_success": {"communication failure": 0.82},  
        },
        "short_reset": {
            "name": "short-range restore factory Settings",
            "cycle": 0.006,
            "cost": 0,
            "fixed_success": {"communication failure": 0.82},  
        },
        "activation": {
            "name": "sensor reactivation",
            "cycle": 0.15,   # 激活频率较低
            "cost": 0,
            "fixed_success": {"communication failure": 0.75},  
        },
    },
    "zone_3": {
        "rts_cts": {
            "name": "RTS/CTS mechanism",
            "cycle": 0.009,  # 安防需要可靠通信
            "cost": 0,
            "prevents": ["hidden node failure"],
        },
        "heartbeat": {
            "name": "node heartbeat monitoring",
            "cycle": 0.004,  # 安防设备需要及时状态更新
            "cost": 0,
            "prevents": ["power failure"],
        },
        "boot_update": {            
            "name": "program/firmware remote update",
            "cycle": 0.005,  # 安防设备需要保持最新安全补丁
            "cost": 0,
            "prevents": ["boot_fault"],
        },
        "wireless_power": {
            "name": "wireless charging module",
            "cycle": 0.006,  # 安防设备功耗中等
            "cost": 0,
            "prevents": ["power failure"],
        },
        "hardware_wai": {
            "name": "peripheral hardware maintenance",
            "cost": 0,
            "reduces": {"hardware failure": 0.85},
        },
        "remote_restart": {
            "name": "remote restart",
            "cycle": 0.005,
            "cost": 0,
            "fixed_success": {"data failure": 0.70, "data loss failure": 0.70}, 
        },
        "remote_reset": {
            "name": "remote restore factory settings",
            "cycle": 0.005,
            "cost": 0,
            "fixed_success": {"data failure": 0.80, "data loss failure": 0.80},  
        },
        "noise": {
            "name": "recalibration of background noise",
            "cycle": 0.10,   # 安防环境需要定期校准避免误报
            "cost": 0,
            "fixed_success": {"data failure": 0.60},  
        },
        "short_restart": {
            "name": "close restart module",
            "cycle": 0.005,
            "cost": 0,
            "fixed_success": {"communication failure": 0.80},  
        },
        "short_reset": {
            "name": "short-range restore factory Settings",
            "cycle": 0.005,
            "cost": 0,
            "fixed_success": {"communication failure": 0.80},  
        },
        "activation": {
            "name": "sensor reactivation",
            "cycle": 0.12,   # 安防设备激活频率中等
            "cost": 0,
            "fixed_success": {"communication failure": 0.70},  
        },
    },    
    "zone_4": {
        "rts_cts": {
            "name": "RTS/CTS mechanism",
            "cycle": 0.015,  # 基础设施监控，通信需求较低
            "cost": 0,       # 低成本设计
            "prevents": ["hidden node failure"],
        },
        "heartbeat": {
            "name": "node heartbeat monitoring",
            "cycle": 0.008,  # 低频心跳，节能设计
            "cost": 0,
            "prevents": ["power failure"],
        },
        "boot_update": {            
            "name": "program/firmware remote update",
            "cycle": 0.008,  # 低频更新
            "cost": 0,
            "prevents": ["boot_fault"],
        },
        "wireless_power": {
            "name": "wireless charging module",
            "cycle": 0.010,  # 超低功耗，充电频率低
            "cost": 0,       # 低成本无线充电
            "prevents": ["power failure"],
        },
        "hardware_wai": {
            "name": "peripheral hardware maintenance",
            "cost": 0,
            "reduces": {"hardware failure": 0.90},  # 基础设施传感器设计更可靠
        },
        "remote_restart": {
            "name": "remote restart",
            "cycle": 0.008,
            "cost": 0,
            "fixed_success": {"data failure": 0.75, "data loss failure": 0.75}, 
        },
        "remote_reset": {
            "name": "remote restore factory settings",
            "cycle": 0.008,
            "cost": 0,
            "fixed_success": {"data failure": 0.85, "data loss failure": 0.85},  
        },
        "noise": {
            "name": "recalibration of background noise",
            "cycle": 0.15,   # 基础设施环境稳定，校准频率低
            "cost": 0,
            "fixed_success": {"data failure": 0.70},  
        },
        "short_restart": {
            "name": "close restart module",
            "cycle": 0.008,
            "cost": 0,
            "fixed_success": {"communication failure": 0.85},  
        },
        "short_reset": {
            "name": "short-range restore factory Settings",
            "cycle": 0.008,
            "cost": 0,
            "fixed_success": {"communication failure": 0.85},  
        },
        "activation": {
            "name": "sensor reactivation",
            "cycle": 0.20,   # 基础设施传感器很少需要重新激活
            "cost": 0,
            "fixed_success": {"communication failure": 0.80},  
        },
    }
}

# 定义可能的故障类型及其月故障概率和对应人工修复成本
FAULT_TYPES = {
    "zone_1": {
        "hidden node failure": {
            "probability": 0.005,  # 工业环境干扰大，隐藏节点故障概率高
            "cost": 600,           # 工业环境维修成本高
        },
        "data failure": {
            "probability": 0.008,  # 振动数据复杂，故障概率较高
            "cost": 550,
        },
        "data loss failure": {
            "probability": 0.007,  # 工业数据传输量大，丢失风险高
            "cost": 580,
        },
        "communication failure": {
            "probability": 0.010,  # 工业环境通信干扰大
            "cost": 520,
        },
        "power failure": {
            "probability": 0.006,  # 工业设备功耗大，电源故障概率高
            "cost": 620,
        },
        "hardware failure": {
            "probability": 0.004,  # 工业环境恶劣，硬件故障概率高
            "cost": 800,           # 工业级硬件更换成本高
        },
        "boot_fault": {
            "probability": 0.002,  # 工业设备启动故障
            "cost": 700,     
        }
    },
    "zone_2": {
        "hidden node failure": {
            "probability": 0.003,  # 环境相对稳定
            "cost": 450,
        },
        "data failure": {
            "probability": 0.004,  # 温湿度数据相对简单
            "cost": 420,
        },
        "data loss failure": {
            "probability": 0.004,  # 数据量小，丢失概率低
            "cost": 430,
        },
        "communication failure": {
            "probability": 0.006,  # 通信环境较好
            "cost": 440,
        },
        "power failure": {
            "probability": 0.003,  # 功耗较低
            "cost": 460,
        },
        "hardware failure": {
            "probability": 0.002,  # 硬件相对可靠
            "cost": 480,
        },
        "boot_fault": {
            "probability": 0.001,  # 启动故障概率低
            "cost": 470,     
        }
    },
    "zone_3": {
        "hidden node failure": {
            "probability": 0.004,  # 安防环境有一定干扰
            "cost": 520,
        },
        "data failure": {
            "probability": 0.006,  # 运动数据较复杂
            "cost": 500,
        },
        "data loss failure": {
            "probability": 0.005,  # 安防数据重要性高，但数据量中等
            "cost": 510,
        },
        "communication failure": {
            "probability": 0.008,  # 安防需要可靠通信
            "cost": 490,
        },
        "power failure": {
            "probability": 0.004,  # 功耗中等
            "cost": 530,
        },
        "hardware failure": {
            "probability": 0.003,  # 硬件可靠性要求高
            "cost": 600,
        },
        "boot_fault": {
            "probability": 0.0015, # 安防设备启动故障
            "cost": 520,     
        }
    },
    "zone_4": {
        "hidden node failure": {
            "probability": 0.002,  # 基础设施环境稳定
            "cost": 380,
        },
        "data failure": {
            "probability": 0.003,  # 结构健康数据相对稳定
            "cost": 350,
        },
        "data loss failure": {
            "probability": 0.002,  # 超低数据丢失率
            "cost": 360,
        },
        "communication failure": {
            "probability": 0.004,  # 通信需求低
            "cost": 370,
        },
        "power failure": {
            "probability": 0.001,  # 超低功耗设计
            "cost": 390,
        },
        "hardware failure": {
            "probability": 0.001,  # 基础设施传感器设计寿命长
            "cost": 420,
        },
        "boot_fault": {
            "probability": 0.0005, # 极低的启动故障率
            "cost": 400,     
        }
    }
}

# 默认参数值（靠近可靠性更高的值）
DEFAULT_PARAM_VALUES = {
    "zone_1": {
        "preventive_check_days": 1,          
        "frequency_heartbeat": 60 / 60,        # 高频心跳
        "heartbeat_loss_threshold": 3           
    },
    "zone_2": {
        "preventive_check_days": 1,            
        "frequency_heartbeat": 300 / 60,        # 中等频率心跳
        "heartbeat_loss_threshold": 3          
    },
    "zone_3": {
        "preventive_check_days": 1,          
        "frequency_heartbeat": 900 / 60,        # 较高频率心跳
        "heartbeat_loss_threshold": 3           
    },
    "zone_4": {
        "preventive_check_days": 1,             
        "frequency_heartbeat": 1800 / 60,       # 低频心跳
        "heartbeat_loss_threshold": 3           
    }
}