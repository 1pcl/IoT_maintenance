# -*- coding: utf-8 -*-
# 定义所有可用维护模块及其属性(未来可以扩展)
MODULES = {
    "rts_cts": {
        "name": "RTS/CTS mechanism",
        "cycle": 0.01,
        "cost": 0,
        "prevents": ["hidden node failure"],
    },
    "heartbeat": {
        "name": "node heartbeat monitoring",
        "cycle": 0.005,
        "cost": 0,
        "prevents": ["power failure"],
    },
    "boot_update": {            
        "name": "program/firmware remote update",
        "cycle": 0.005,
        "cost": 0,
        "prevents": ["boot_fault"],
    },
    "wireless_power": {
        "name": "wireless charging module",
        "cycle": 0.006,
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
        "fixed_success": {"data failure": 0.7, "data loss failure": 0.7}, 
    },
    "remote_reset": {
        "name": "remote restore factory settings",
        "cycle": 0.005,
        "cost": 0,
        "fixed_success": {"data failure": 0.8, "data loss failure": 0.8},  
    },
    "noise": {
        "name": "recalibration of background noise",
        "cycle": 0.1,
        "cost": 0,
        "fixed_success": {"data failure": 0.6},  
    },
    "short_restart": {
        "name": "close restart module",
        "cycle": 0.005,
        "cost": 0,
        "fixed_success": {"communication failure": 0.8},  
    },
    "short_reset": {
        "name": "short-range restore factory Settings",
        "cycle": 0.005,
        "cost": 0,
        "fixed_success": {"communication failure": 0.8},  
    },
    "activation": {
        "name": "sensor reactivation",
        "cycle": 0.12,
        "cost": 0,
        "fixed_success": {"communication failure": 0.7},  
    },
}

# 定义可能的故障类型及其月故障概率和对应人工修复成本
FAULT_TYPES = {
    "hidden node failure": {
        "probability": 0.003,
        "cost": 500,
    },
    "data failure": {
        "probability": 0.005,
        "cost": 500,
    },
    "data loss failure": {
        "probability": 0.005,
        "cost": 500,
    },
    "communication failure": {
        "probability": 0.008,
        "cost": 500,
    },
    "power failure": {
        "probability": 0,
        "cost": 500,
    },
    "hardware failure": {
        "probability": 0.001,
        "cost": 510,
    },
    "boot_fault": {
        "probability": 0.0005,
        "cost": 500,     
    }
}

# 默认参数值（靠近可靠性更高的值）
DEFAULT_PARAM_VALUES = {
    "warning_energy": 50.0,
    "preventive_check_days": 1,
    "frequency_heartbeat": 900 / 60,  # 默认值为动物房的场景（该场景的频率较高），会根据输入的场景调节（在生成场景中会修改）越小可靠性越高
    "heartbeat_loss_threshold": 3
}