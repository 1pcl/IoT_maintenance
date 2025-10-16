# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from matplotlib.lines import Line2D
from utilities import FAULT_TYPES, MODULES, DEFAULT_PARAM_VALUES
from scene_generator import SceneGenerator

class WSNSimulation:
    def __init__(self, scene_data, selected_modules_by_zone, simulation_params_by_zone=None):
        # 从场景数据中提取配置、节点和基站
        self.config = scene_data["config"]
        self.nodes_data = scene_data["nodes"]
        self.base_station_data = scene_data["base_station"]
        self.zone_configs = scene_data["zone_configs"]
        self.application = self.config.get("application")
        
        # 初始化其他参数 - 使用按区域配置
        self.selected_modules_by_zone = selected_modules_by_zone
        self.simulation_params_by_zone = simulation_params_by_zone if simulation_params_by_zone else {}
        
        # 初始化网络参数
        self._initialize_network_parameters()
        
        # 初始化网络状态
        self._initialize_network_state()

    def _initialize_network_parameters(self):
        """初始化网络参数"""
        self.area = self.config["area_size"]
        self.sensor_num = self.config["sensor_num"]
        self.life_span = self.config["life_span"]
        self.month_time = 30 * 24 * 60 * 60.0  # 一个月的秒数

        self.configuring_package_size = self.config["configuring_package_size"]
        self.TDMA_package_size = self.config["TDMA_package_size"]
        
        # 多跳参数
        self.intra_cluster_multihop = self.config["intra_cluster_multihop"]
        self.inter_cluster_multihop = self.config["inter_cluster_multihop"]
        
        # RTS/CTS参数
        self.rts_cts_size = self.config["rts_cts_size"]

    def _initialize_network_state(self):
        """初始化网络状态"""
        self.current_month = 0
        self.total_months = self.life_span
        self.preventive_maintenance_cost = 0
        
        # 创建基站 - 确保基站ID为0
        self.base_station = self.base_station_data.copy()
        self.base_station["id"] = 0  # 统一使用0作为基站ID
        
        # 创建节点
        self.nodes = []
        for node_data in self.nodes_data:
            zone_name = node_data["zone"]
            # 正确初始化故障计时器
            fault_timers = {}
            for fault_type in FAULT_TYPES[zone_name]:
                fault_timers[fault_type] = {
                    "next_time": -1,
                    "count": 0,
                    "last_fault_round": -1,
                    "fault_flag": 0,
                    "fault_round": 0
                }

            node = {
                "id": node_data["id"],
                "energy": self.zone_configs[zone_name]["initial_energy"],
                "x": node_data["x"],
                "y": node_data["y"],
                "distance": self._distance(node_data["x"], node_data["y"], 
                                         self.base_station["x"], self.base_station["y"]),
                "cluster_head": False,
                "routing_information": [],
                "cluster_info": [],
                "sector": 0,
                "fault": False,
                "fault_timers": fault_timers,
                "zone": zone_name,
            }
            
            self.nodes.append(node)
        
        # 为每个区域初始化扇区划分
        self._initial_sector_division()

    def _distance(self, x1, y1, x2, y2):
        """计算两点之间的距离"""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _initial_sector_division(self):
        """初始扇区划分"""
        # 按区域分组节点
        zone_nodes = {}
        for node in self.nodes:
            zone = node["zone"]
            if zone not in zone_nodes:
                zone_nodes[zone] = []
            zone_nodes[zone].append(node)
        
        # 为每个区域独立进行扇区划分
        for zone_name, nodes in zone_nodes.items():
            if self.inter_cluster_multihop and nodes:
                # 收集所有节点到基站的距离
                distanciasBS = []
                for node in nodes:
                    dist = self._distance(node["x"], node["y"], 
                                        self.base_station["x"], self.base_station["y"])
                    distanciasBS.append(dist)
                    node["distance"] = dist  # 更新节点距离

                # 为每个节点分配扇区号
                for node in nodes:
                    dist = self._distance(node["x"], node["y"], 
                                        self.base_station["x"], self.base_station["y"])
                    node["sector"] = self._setorizacaoCH(distanciasBS, dist, 
                                                        self.zone_configs[zone_name]["max_hops"])

    def _setorizacaoCH(self, listaDistancias, distancia, divisor):
        """扇区划分：用于多跳路由的扇区划分(簇间)"""
        if not listaDistancias:
            return 1
            
        menor = min(listaDistancias)
        maior = max(listaDistancias)
        
        if divisor == 0:
            divisor = 1
        valor = (maior - menor) / divisor

        for i in range(1, divisor + 1):
            if distancia <= menor + i * valor:
                return i
        return divisor

    def _menorLista(self, lista):
        """求最小值"""
        return min(lista) if lista else 0

    def _maiorLista(self, lista):
        """求最大值"""
        return max(lista) if lista else 0

    def _energy_consumption_tx(self, energy, distance, packet_size, config):
        """计算发送能耗"""
        energy_cost = (config["energy_consumption_tx"] * packet_size * 8 + 
                      config["energy_amplifier"] * packet_size * 8 * (distance ** 2))
        return energy - energy_cost
    
    def _energy_consumption_rx(self, energy, packet_size, config):
        """计算接收能耗"""
        energy_cost = config["energy_consumption_rx"] * packet_size * 8
        return energy - energy_cost
    
    def _energy_consumption_agg(self, energy, packet_size, num_packets, config):
        """计算数据聚合能耗"""
        energy_cost = config["energy_aggregation"] * packet_size * 8 * num_packets
        return energy - energy_cost
    
    def _setorizacao(self, lista, divisor, config):
        """簇内扇区划分"""
        if not lista:
            return lista
            
        # 提取距离并排序
        distances = [node[3] for node in lista]
        distances.sort()
        
        if divisor == 0:
            divisor = 1
        valor = (distances[-1] - distances[0]) / divisor
        
        # 扇区划分
        for node in lista:
            for i in range(1, divisor + 1):
                if node[3] <= distances[0] + i * valor:
                    node[4] = i
                    break
            else:
                node[4] = divisor
                
        return lista
    
    def _ajuste_alcance_nodeCH(self, CH, config):
        """调整簇头通信范围"""
        for nodeCH in CH:
            max_distance = 0
            # 检查簇内所有节点，找到最远距离
            for node in nodeCH["cluster_info"]:
                if node[3] > max_distance:
                    max_distance = node[3]
            # 设置簇头通信范围为最远节点距离
            nodeCH["distance"] = max_distance
    
    def _contEncaminhamento(self, id, listaID):
        """统计路由表中特定ID出现的次数"""
        return listaID.count(id)
    
    def _localizaObjetoCH(self, id, nodes):
        """根据ID查找节点"""
        for node in nodes:
            if node["id"] == id:
                return node
        return None
    
    def _verifica_eleitos(self, nodes):
        """检查是否所有节点都被选为簇头"""
        cluster_head_count = sum(1 for node in nodes if node["cluster_head"])
        return cluster_head_count == len(nodes)
    
    def _select_ch_for_zone(self, zone_nodes, total_round_num, zone_config):
        """为特定区域选择簇头"""
        if not zone_nodes:
            return [], []
            
        # 重置该区域所有节点的路由信息
        for node in zone_nodes:
            if not node["fault"]:
                node["routing_information"] = []
                node["cluster_info"] = []

        # 定期重置簇头选择
        cluster_head_change_round = round(1.0 / zone_config["cluster_head_percentage"])
        if total_round_num % cluster_head_change_round == 0:
            for node in zone_nodes:
                node["cluster_head"] = False
        
        # 清空簇结构和非簇结构
        cluster_heads = []
        non_cluster_heads = []

        # 计算选择阈值
        limiar = (zone_config["cluster_head_percentage"] / 
                 (1.0 - zone_config["cluster_head_percentage"] * 
                  (total_round_num % cluster_head_change_round)))

        # 选择簇头
        for node in zone_nodes:
            if not node["fault"]:
                rand = random.random()
                if rand < limiar and not node["cluster_head"]:
                    node["cluster_head"] = True
                    cluster_heads.append(node)
                else:
                    non_cluster_heads.append(node)
        
        # 确保必须有一个簇头
        if not cluster_heads:
            for node in zone_nodes:
                if not node["fault"] and not node["cluster_head"]:
                    node["cluster_head"] = True
                    cluster_heads.append(node)
                    if node in non_cluster_heads:
                        non_cluster_heads.remove(node)
                    break
        
        # 如果还是没有簇头，选择第一个可用节点
        if not cluster_heads:
            for node in zone_nodes:
                if not node["fault"]:
                    node["cluster_head"] = True
                    cluster_heads.append(node)
                    if node in non_cluster_heads:
                        non_cluster_heads.remove(node)
                    break

        if cluster_heads:
            # 簇形成阶段
            # 簇头发送广播
            broadcast_packets = []
            for ch in cluster_heads:
                # 基站路由信息 - 使用ID 0表示基站
                base_station_info = [0, self.base_station["x"], self.base_station["y"], 
                                   ch["distance"], self.base_station.get("sector", 0)]
                ch["routing_information"].append(base_station_info)
                
                # 创建广播包
                broadcast_info = [ch["id"], ch["x"], ch["y"], 0.0, ch["sector"]]
                broadcast_packets.append(broadcast_info)
                
                # 发送广播能耗
                ch["energy"] = self._energy_consumption_tx(ch["energy"], ch["distance"], 
                                                          self.configuring_package_size, zone_config)

            # 簇头接收其他簇头广播
            for ch in cluster_heads:
                for packet in broadcast_packets:
                    if packet[0] != ch["id"]:  # 不接收自己的广播
                        ch["routing_information"].append(packet)
                        ch["energy"] = self._energy_consumption_rx(ch["energy"], 
                                                                  self.configuring_package_size, zone_config)

            # 普通节点选择簇头
            if non_cluster_heads:
                # 普通节点接收簇头广播并选择最近的簇头
                for node in non_cluster_heads:
                    min_distance = float('inf')
                    selected_ch_info = None
                    
                    for ch_packet in broadcast_packets:
                        # 接收广播能耗
                        node["energy"] = self._energy_consumption_rx(node["energy"], 
                                                                     self.configuring_package_size, zone_config)
                        
                        # 计算距离
                        dist = self._distance(node["x"], node["y"], 
                                            ch_packet[1], ch_packet[2])
                        if dist < min_distance:
                            min_distance = dist
                            selected_ch_info = ch_packet
                    
                    # 更新节点的路由信息
                    if selected_ch_info:
                        node["routing_information"] = [selected_ch_info]
                        node["distance"] = min_distance
                
                # 普通节点发送加入请求
                for node in non_cluster_heads:
                    if node["routing_information"]:
                        node_info = [node["id"], node["x"], node["y"], node["distance"], 0]
                        ch_id = node["routing_information"][0][0]
                        
                        # 找到对应的簇头并添加节点信息
                        for ch in cluster_heads:
                            if ch["id"] == ch_id:
                                ch["cluster_info"].append(node_info)
                                break
                        
                        # 发送加入请求能耗
                        node["energy"] = self._energy_consumption_tx(node["energy"], node["distance"], 
                                                                     self.configuring_package_size, zone_config)
                
                # 簇头接收加入请求
                for ch in cluster_heads:
                    for _ in range(len(ch["cluster_info"])):
                        ch["energy"] = self._energy_consumption_rx(ch["energy"], 
                                                                  self.configuring_package_size, zone_config)

                # 簇头发送TDMA调度表
                self._ajuste_alcance_nodeCH(cluster_heads, zone_config)
                clusters = []
                for ch in cluster_heads:
                    clustered_nodes = self._setorizacao(ch["cluster_info"], 
                                                       zone_config["max_hops"], zone_config)
                    clusters.append([ch["id"], clustered_nodes])
                    ch["energy"] = self._energy_consumption_tx(ch["energy"], ch["distance"], 
                                                              self.TDMA_package_size, zone_config)
                
                # 普通节点接收TDMA表
                for node in non_cluster_heads:
                    if node["routing_information"]:
                        ch_id = node["routing_information"][0][0]
                        # 找到对应的簇分配信息
                        for cluster in clusters:
                            if cluster[0] == ch_id:
                                node["cluster_info"] = cluster[1]
                                break
                    
                    node["energy"] = self._energy_consumption_rx(node["energy"], 
                                                                self.TDMA_package_size, zone_config)

                # 更新簇头到基站的距离
                for ch in cluster_heads:
                    ch["distance"] = self._distance(ch["x"], ch["y"], 
                                                  self.base_station["x"], self.base_station["y"])

                # 簇内多跳路由
                if self.intra_cluster_multihop:
                    for node in non_cluster_heads:
                        if not node["cluster_info"]:
                            continue
                            
                        # 找到节点在簇内的扇区
                        node_sector = 0
                        for cluster_node in node["cluster_info"]:
                            if node["id"] == cluster_node[0]:
                                node_sector = cluster_node[4]
                                break
                        
                        # 寻找更靠近簇头的邻居节点
                        relay_id = node["routing_information"][0][0]  # 默认是簇头
                        min_distance = node["distance"]
                        
                        for cluster_node in node["cluster_info"]:
                            dist = self._distance(node["x"], node["y"], 
                                                cluster_node[1], cluster_node[2])
                            if dist < min_distance and cluster_node[4] < node_sector:
                                relay_id = cluster_node[0]
                                min_distance = dist
                        
                        # 更新路由信息
                        for cluster_node in node["cluster_info"]:
                            if cluster_node[0] == relay_id:
                                node["routing_information"] = [cluster_node]
                                node["distance"] = min_distance
                                break
                
                # 簇间多跳路由
                if self.inter_cluster_multihop:
                    for ch in cluster_heads:
                        min_distance = ch["distance"]  # 到基站的距离
                        best_route = None
                        
                        # 寻找更靠近基站的簇头作为中继
                        for route in ch["routing_information"]:
                            if isinstance(route, list) and len(route) > 4:
                                dist_to_route = self._distance(ch["x"], ch["y"], route[1], route[2])
                                if dist_to_route < min_distance and route[4] < ch["sector"]:
                                    min_distance = dist_to_route
                                    best_route = route
                        
                        if best_route:
                            ch["routing_information"] = [best_route]
                            ch["distance"] = min_distance
        
        return cluster_heads, non_cluster_heads

    def _simulate_round_for_zone(self, cluster_heads, non_cluster_heads, total_round_num, zone_config, zone_name):
        """为特定区域模拟一轮网络数据传输运行"""
        if not cluster_heads:
            return 0
            
        selected_modules = self.selected_modules_by_zone.get(zone_name, [])
        sim_params = self.simulation_params_by_zone.get(zone_name, {})
        
        # 计算区域特定的轮次参数
        frames_per_round = zone_config["frames_per_round"]
        frequency_sampling = zone_config["frequency_sampling"]
        rounds_time = frequency_sampling * frames_per_round
        
        # RTS/CTS参数
        rts_cts_flag = 1 if "rts_cts" in selected_modules else 0
        
        # 心跳相关参数
        if "heartbeat" in selected_modules:
            frequency_heartbeat = sim_params.get("frequency_heartbeat", 
                                                DEFAULT_PARAM_VALUES[zone_name]["frequency_heartbeat"])
            heartbeat_count_per_sampling = frequency_heartbeat / frequency_sampling
        else:
            heartbeat_count_per_sampling = 0
        
        # 预防性维护检查轮数
        preventive_check_days = sim_params.get("preventive_check_days", 
                                             DEFAULT_PARAM_VALUES[zone_name]["preventive_check_days"])
        check_round = round(preventive_check_days * 24 * 60 * 60 / rounds_time) if rounds_time > 0 else 1

        # 创建路由映射表
        routing_map = []
        for node in non_cluster_heads:
            if node["routing_information"]:
                routing_map.append(node["routing_information"][0][0])

        # 模拟数据传输
        loss_data_count = 0
        actual_transmission = 0
        confirmed_frames = 0
        
        for frame_count in range(frames_per_round):
            # 普通节点发送数据
            for node in non_cluster_heads:
                if node["fault"]:
                    continue
                    
                # 采样能耗
                node["energy"] -= zone_config["energy_sampling"]
                
                # 心跳能耗
                if heartbeat_count_per_sampling > 0:
                    node["energy"] = self._energy_consumption_tx(node["energy"], node["distance"], 
                                                               zone_config["heartbeat_packet_size"] * heartbeat_count_per_sampling, zone_config)
                
                # 数据聚合能耗（仅簇内多跳）
                if self.intra_cluster_multihop:
                    relay_count = self._contEncaminhamento(node["id"], routing_map)
                    if relay_count > 0:
                        node["energy"] = self._energy_consumption_agg(node["energy"], zone_config["packet_size"], 
                                                                      (relay_count + 1), zone_config)
                
                # RTS/CTS握手
                if rts_cts_flag:
                    node["energy"] = self._energy_consumption_tx(node["energy"], node["distance"], 
                                                               self.rts_cts_size, zone_config)
                    node["energy"] = self._energy_consumption_rx(node["energy"], self.rts_cts_size, zone_config)
                
                # 发送数据
                node["energy"] = self._energy_consumption_tx(node["energy"], node["distance"], 
                                                           zone_config["packet_size"], zone_config)
                actual_transmission += 1
                
                if node["energy"] > 0:
                    confirmed_frames += 1
                else:
                    if "wireless_power" in selected_modules:
                        confirmed_frames += 1
                        node["energy"] += zone_config["initial_energy"]
                    else:
                        node["fault"] = True
            
            # 簇头接收数据
            for ch in cluster_heads:
                if ch["fault"]:
                    continue
                    
                relay_count = self._contEncaminhamento(ch["id"], routing_map)
                for _ in range(relay_count):
                    if rts_cts_flag:
                        ch["energy"] = self._energy_consumption_rx(ch["energy"], self.rts_cts_size, zone_config)
                        ch["energy"] = self._energy_consumption_tx(ch["energy"], ch["distance"], 
                                                                 self.rts_cts_size, zone_config)
                    ch["energy"] = self._energy_consumption_rx(ch["energy"], zone_config["packet_size"], zone_config)
            
            # 普通节点接收数据（仅簇内多跳）
            if self.intra_cluster_multihop:
                for node in non_cluster_heads:
                    if node["fault"]:
                        continue
                        
                    relay_count = self._contEncaminhamento(node["id"], routing_map)
                    for _ in range(relay_count):
                        if rts_cts_flag:
                            node["energy"] = self._energy_consumption_rx(node["energy"], self.rts_cts_size, zone_config)
                            node["energy"] = self._energy_consumption_tx(node["energy"], node["distance"], 
                                                                       self.rts_cts_size, zone_config)
                        node["energy"] = self._energy_consumption_rx(node["energy"], zone_config["packet_size"], zone_config)
            
            # 簇头向基站发送数据
            for ch in cluster_heads:
                if ch["fault"]:
                    continue
                    
                # 采样和心跳
                ch["energy"] -= zone_config["energy_sampling"]
                if heartbeat_count_per_sampling > 0:
                    ch["energy"] = self._energy_consumption_tx(ch["energy"], ch["distance"], 
                                                             zone_config["heartbeat_packet_size"] * heartbeat_count_per_sampling, zone_config)
                
                # 数据聚合
                relay_count = self._contEncaminhamento(ch["id"], routing_map)
                if relay_count > 0:
                    ch["energy"] = self._energy_consumption_agg(ch["energy"], zone_config["packet_size"], 
                                                              (relay_count + 1), zone_config)
                
                # 多跳传输到基站
                current_node = ch
                if not current_node["routing_information"]:
                    continue
                    
                next_hop_id = current_node["routing_information"][0][0]
                
                while next_hop_id != 0:  # 0表示基站
                    if rts_cts_flag:
                        current_node["energy"] = self._energy_consumption_tx(current_node["energy"], current_node["distance"], 
                                                                           self.rts_cts_size, zone_config)
                        current_node["energy"] = self._energy_consumption_rx(current_node["energy"], self.rts_cts_size, zone_config)
                    
                    current_node["energy"] = self._energy_consumption_tx(current_node["energy"], current_node["distance"], 
                                                                       zone_config["packet_size"], zone_config)
                    actual_transmission += 1
                    
                    if current_node["energy"] > 0:
                        confirmed_frames += 1
                    else:
                        if "wireless_power" in selected_modules:
                            confirmed_frames += 1
                            current_node["energy"] += zone_config["initial_energy"]
                        else:
                            current_node["fault"] = True
                            break
                    
                    # 查找下一跳节点
                    next_node = None
                    for node in cluster_heads + non_cluster_heads:
                        if node["id"] == next_hop_id and not node["fault"]:
                            next_node = node
                            break
                    
                    if not next_node:
                        break
                        
                    # 下一跳节点接收数据
                    if rts_cts_flag:
                        next_node["energy"] = self._energy_consumption_rx(next_node["energy"], 
                                                                         self.rts_cts_size, zone_config)
                        next_node["energy"] = self._energy_consumption_tx(next_node["energy"], 
                                                                         next_node["distance"], self.rts_cts_size, zone_config)
                    next_node["energy"] = self._energy_consumption_rx(next_node["energy"], 
                                                                     zone_config["packet_size"], zone_config)
                    
                    current_node = next_node
                    if not current_node["routing_information"]:
                        break
                    next_hop_id = current_node["routing_information"][0][0]
        
        loss_data_count = actual_transmission - confirmed_frames
        
        # 定期维护检查
        if check_round > 0 and total_round_num % check_round == 0:
            self.fixing_fault_for_zone(zone_name, cluster_heads + non_cluster_heads, total_round_num)
        
        return loss_data_count

    def plot_cluster_topology(self, month, round_num, save_folder="cluster_visualizations"):
        """绘制簇头拓扑图，显示簇头、节点和连接关系"""
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        plt.figure(figsize=(14, 10))
        
        # 绘制区域分割线
        area_size = self.config["area_size"]
        half_size = area_size / 2
        plt.axvline(x=half_size, color='black', linestyle='--', linewidth=2, alpha=0.7)
        plt.axhline(y=half_size, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        # 绘制区域背景
        zones = {
            "zone_1": {"x_range": (0, half_size), "y_range": (half_size, area_size), "color": "red", "alpha": 0.1},
            "zone_2": {"x_range": (half_size, area_size), "y_range": (half_size, area_size), "color": "blue", "alpha": 0.1},
            "zone_3": {"x_range": (half_size, area_size), "y_range": (0, half_size), "color": "green", "alpha": 0.1},
            "zone_4": {"x_range": (0, half_size), "y_range": (0, half_size), "color": "orange", "alpha": 0.1}
        }
        
        for zone_name, zone_info in zones.items():
            rect = plt.Rectangle(
                (zone_info["x_range"][0], zone_info["y_range"][0]),
                zone_info["x_range"][1] - zone_info["x_range"][0],
                zone_info["y_range"][1] - zone_info["y_range"][0],
                fill=True,
                color=zone_info["color"],
                alpha=zone_info["alpha"]
            )
            plt.gca().add_patch(rect)
        
        # 按区域绘制节点和连接
        legend_elements = []
        zone_colors = {"zone_1": "red", "zone_2": "blue", "zone_3": "green", "zone_4": "orange"}
        
        # 跟踪已绘制的连接，避免重复
        drawn_connections = set()
        
        for zone_name in self.zone_configs.keys():
            zone_nodes = [node for node in self.nodes if node["zone"] == zone_name]
            cluster_heads = [node for node in zone_nodes if node["cluster_head"] and not node["fault"]]
            non_cluster_heads = [node for node in zone_nodes if not node["cluster_head"] and not node["fault"]]
            faulty_nodes = [node for node in zone_nodes if node["fault"]]
            
            color = zone_colors[zone_name]
            
            # 绘制故障节点
            if faulty_nodes:
                faulty_x = [node["x"] for node in faulty_nodes]
                faulty_y = [node["y"] for node in faulty_nodes]
                plt.scatter(faulty_x, faulty_y, color='gray', s=50, alpha=0.5, marker='x')
            
            # 绘制普通节点
            if non_cluster_heads:
                non_ch_x = [node["x"] for node in non_cluster_heads]
                non_ch_y = [node["y"] for node in non_cluster_heads]
                plt.scatter(non_ch_x, non_ch_y, color=color, s=50, alpha=0.7, marker='o')
            
            # 绘制簇头节点
            if cluster_heads:
                ch_x = [node["x"] for node in cluster_heads]
                ch_y = [node["y"] for node in cluster_heads]
                plt.scatter(ch_x, ch_y, color=color, s=150, alpha=1.0, marker='*', edgecolors='black', linewidths=1)
            
            # 绘制连接关系 - 修复基站连接问题
            for node in zone_nodes:
                if node["fault"]:
                    continue
                    
                if node["routing_information"]:
                    next_hop_info = node["routing_information"][0]
                    if isinstance(next_hop_info, list) and len(next_hop_info) > 0:
                        next_hop_id = next_hop_info[0]
                        
                        # 生成连接标识符避免重复
                        connection_id = (min(node["id"], next_hop_id), max(node["id"], next_hop_id))
                        if connection_id in drawn_connections:
                            continue
                        drawn_connections.add(connection_id)
                        
                        # 连接到基站
                        if next_hop_id == 0:
                            plt.plot([node["x"], self.base_station["x"]], 
                                    [node["y"], self.base_station["y"]], 
                                    color=color, linewidth=2, alpha=0.7, linestyle='-')
                        else:
                            # 连接到其他节点
                            next_node = self._localizaObjetoCH(next_hop_id, self.nodes)
                            if next_node and not next_node["fault"]:
                                plt.plot([node["x"], next_node["x"]], 
                                        [node["y"], next_node["y"]], 
                                        color=color, linewidth=1, alpha=0.5)
            
            # 添加图例元素
            if cluster_heads:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color, 
                             markersize=12, label=f'{zone_name} Cluster Heads ({len(cluster_heads)})')
                )
            if non_cluster_heads:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                             markersize=8, label=f'{zone_name} Nodes ({len(non_cluster_heads)})', alpha=0.7)
                )
        
        # 绘制基站
        plt.plot(self.base_station["x"], self.base_station["y"], 's', 
                color='purple', markersize=20, label='Base Station', markeredgecolor='black', markeredgewidth=2)
        legend_elements.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', 
                     markersize=12, label='Base Station', markeredgecolor='black', markeredgewidth=1)
        )
        
        # 设置图例和标题
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=18)
        # plt.title(f'Cluster Topology - Month {month}, Round {round_num}\n'
        #          f'LEACH Protocol with Multi-hop Routing', fontsize=16, pad=20)
        
        plt.xlim(0, area_size)
        plt.ylim(0, area_size)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        # 保存图片
        save_path = os.path.join(save_folder, f'cluster_topology_month_{month}_round_{round_num}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cluster topology saved: {save_path}")
        return save_path

    def run_simulation(self):
        """运行完整的仿真 - 按月仿真，每轮独立处理四个区域"""
        # 初始化数据结构 - 按区域分开
        total_loss_data_count = {}
        all_node_fault_list = {}
        total_check_count = {}
        
        # 初始化每个区域的数据结构
        for zone_name in self.zone_configs.keys():
            total_loss_data_count[zone_name] = 0
            all_node_fault_list[zone_name] = []
            # 计算每个区域的检查次数
            sim_params = self.simulation_params_by_zone.get(zone_name, {})
            preventive_check_days = sim_params.get("preventive_check_days", 
                                        DEFAULT_PARAM_VALUES[zone_name]["preventive_check_days"])
            total_check_count[zone_name] = self.life_span * 30.0 / preventive_check_days
        
        # 按月运行仿真
        for month in range(1, self.total_months + 1):
            self.current_month = month
            # print(f"Simulating month {month}/{self.total_months}")
            
            # 为每个区域计算本月的轮次数
            zone_rounds = {}
            for zone_name, zone_config in self.zone_configs.items():
                frequency_sampling = zone_config["frequency_sampling"]
                frames_per_round = zone_config["frames_per_round"]
                round_time = frequency_sampling * frames_per_round  # 一轮的时间（秒）
                rounds_per_month = int(self.month_time / round_time) if round_time > 0 else 1
                zone_rounds[zone_name] = rounds_per_month
            
            # 找到最大轮次数
            max_rounds = max(zone_rounds.values()) if zone_rounds else 1
            
            # 初始化本月损失计数器 - 按区域分开
            monthly_loss_count = {zone_name: 0 for zone_name in self.zone_configs.keys()}
            
            # 按轮次运行
            for round_num in range(1, max_rounds + 1):
                # 为每个区域处理当前轮次
                for zone_name, zone_config in self.zone_configs.items():
                    # 如果该区域有当前轮次
                    if round_num <= zone_rounds[zone_name]:
                        zone_nodes = [node for node in self.nodes if node["zone"] == zone_name]
                        
                        # 选择簇头
                        cluster_heads, non_cluster_heads = self._select_ch_for_zone(
                            zone_nodes, round_num, zone_config)
                        
                        # 模拟故障
                        self.simulation_fault_for_zone(zone_nodes, round_num, zone_rounds[zone_name], zone_config)
                        
                        # 模拟一轮网络运行
                        loss_data_count = self._simulate_round_for_zone(
                            cluster_heads, non_cluster_heads, round_num, zone_config, zone_name)
                        
                        # 更新该区域的月度损失计数
                        monthly_loss_count[zone_name] += loss_data_count
                        
                        # 电量故障修复
                        self.simulation_power_failure_for_zone(zone_nodes, round_num)
                
                # # 在第一个月第一轮后绘制拓扑图
                # if month == 1 and round_num == 1:
                #     self.plot_cluster_topology(month, round_num)
            
            # 本月结束后，将本月损失累加到总损失中
            for zone_name in self.zone_configs.keys():
                total_loss_data_count[zone_name] += monthly_loss_count[zone_name]
        
        # 收集所有节点的故障信息 - 按区域分类
        for zone_name in self.zone_configs.keys():
            all_node_fault_list[zone_name] = []
            zone_nodes = [node for node in self.nodes if node["zone"] == zone_name]
            for node in zone_nodes:
                # 只收集该区域对应的故障类型
                zone_fault_timers = {}
                for fault_type in FAULT_TYPES[zone_name]:
                    if fault_type in node["fault_timers"]:
                        zone_fault_timers[fault_type] = node["fault_timers"][fault_type]
                all_node_fault_list[zone_name].append({
                    "node_id": node["id"],
                    "fault_timers": zone_fault_timers
                })
        
        return total_loss_data_count, all_node_fault_list, total_check_count

    def simulation_power_failure_for_zone(self, zone_nodes, current_round_num):
        """为特定区域模拟电量故障修复"""
        for node in zone_nodes:
            zone_name = node["zone"]
            selected_modules = self.selected_modules_by_zone.get(zone_name, [])
            if "wireless_power" in selected_modules and node["energy"] <= 0:
                config = self.zone_configs[zone_name]
                node["energy"] += config["initial_energy"]

    def simulation_fault_for_zone(self, zone_nodes, current_round_num, rounds_per_month, zone_config):
        """为特定区域模拟故障"""
        if not zone_nodes:
            return
            
        zone_name = zone_nodes[0]["zone"]
        zone_fault_types = FAULT_TYPES[zone_name]
        selected_modules = self.selected_modules_by_zone.get(zone_name, [])
        
        for node in zone_nodes:
            for fault_type, params in zone_fault_types.items():
                timer = node["fault_timers"][fault_type]
                raw_prob = params['probability']
                
                # 应用模块效果
                prevented = any(fault_type in MODULES[zone_name][module].get("prevents", []) 
                              for module in selected_modules if module in MODULES[zone_name])
                if prevented or raw_prob <= 0:
                    continue
                
                # 减少效果
                reduction = 0
                for module in selected_modules:
                    if module in MODULES[zone_name]:
                        if "reduces" in MODULES[zone_name][module] and fault_type in MODULES[zone_name][module]["reduces"]:
                            reduction = max(reduction, MODULES[zone_name][module]["reduces"][fault_type])
                
                if reduction > 0:
                    raw_prob = params['probability'] * (1 - reduction)
                
                if raw_prob > 0:
                    if timer["next_time"] == -1:
                        lambda_fail = -math.log(1 - raw_prob) / rounds_per_month
                        timer["next_time"] = current_round_num + round(random.expovariate(lambda_fail))
                    
                    if timer["next_time"] == current_round_num:
                        if not self.attempt_repair(node, fault_type):
                            timer["count"] += 1
                            node["fault"] = True
                            timer["last_fault_round"] = current_round_num
                            timer["fault_flag"] = 1
                        
                        lambda_fail = -math.log(1 - raw_prob) / rounds_per_month
                        timer["next_time"] = current_round_num + round(random.expovariate(lambda_fail))

    def fixing_fault_for_zone(self, zone_name, nodes, current_round_num):
        """为特定区域修复故障"""
        for node in nodes:
            if node["fault"]:
                selected_modules = self.selected_modules_by_zone.get(zone_name, [])
                sim_params = self.simulation_params_by_zone.get(zone_name, {})
                config = self.zone_configs[zone_name]
                
                for fault_type in FAULT_TYPES[zone_name]:
                    timer = node["fault_timers"][fault_type]
                    if timer["fault_flag"] == 1:
                        frames_per_round = config["frames_per_round"]
                        frequency_sampling = config["frequency_sampling"]
                        rounds_time = frequency_sampling * frames_per_round
                        preventive_check_days = sim_params.get("preventive_check_days", 
                                                             DEFAULT_PARAM_VALUES[zone_name]["preventive_check_days"])
                        check_round = round(preventive_check_days * 24 * 60 * 60 / rounds_time) if rounds_time > 0 else 1
                        
                        heartbeat_loss_threshold = sim_params.get("heartbeat_loss_threshold", 
                                                                DEFAULT_PARAM_VALUES[zone_name]["heartbeat_loss_threshold"])
                        
                        if (check_round > 0 and current_round_num % check_round == 0) or \
                           ("heartbeat" in selected_modules and 
                            current_round_num - timer["last_fault_round"] >= heartbeat_loss_threshold):
                            node["fault"] = False
                            timer["fault_flag"] = 0
                            timer["fault_round"] += current_round_num - timer["last_fault_round"]

    def get_repair_modules(self, node, fault_type):
        """获取可用于修复指定故障的模块列表"""
        zone_name = node["zone"]
        selected_modules = self.selected_modules_by_zone.get(zone_name, [])
        repair_modules = []
        
        for module_name in selected_modules:
            if module_name in MODULES[zone_name]:
                module = MODULES[zone_name][module_name]
                if 'fixed_success' in module and fault_type in module['fixed_success']:
                    repair_modules.append({
                        'name': module_name,
                        'success_rate': module['fixed_success'][fault_type]
                    })
        return repair_modules

    def attempt_repair(self, node, fault_type):
        """尝试分级维护修复故障"""
        repair_modules = self.get_repair_modules(node, fault_type)
        result = False
        
        if not repair_modules:
            return result

        # 尝试维护手段
        for i, repair_module in enumerate(repair_modules):
            if i >= 3:  # 最多尝试三个维护手段
                break
            rand = random.random()
            if rand < repair_module['success_rate']:
                result = self.maintenance_modules_energy(node, repair_module['name'])
                if result:
                    break
        
        return result

    def maintenance_modules_energy(self, node, module):
        """执行维护模块能耗"""
        result = True
        zone_name = node["zone"]
        config = self.zone_configs[zone_name]
        selected_modules = self.selected_modules_by_zone.get(zone_name, [])
        rts_cts_flag = 1 if "rts_cts" in selected_modules else 0
        
        if module in ["remote_restart", "remote_reset", "noise"]:
            command_size = config["maintenance_noise_size"] if module == "noise" else config["maintenance_instruction_size"]
            result = self.send_command_to_node(node["id"], command_size)
        elif module in ["short_restart", "short_reset", "activation"]:
            if rts_cts_flag == 1:
                node["energy"] = self._energy_consumption_rx(node["energy"], self.rts_cts_size, config)
                node["energy"] = self._energy_consumption_tx(node["energy"], node["distance"], self.rts_cts_size, config)
            node["energy"] = self._energy_consumption_rx(node["energy"], config["packet_size"], config)
            result = True
        
        return result

    def send_command_to_node(self, target_node_id, command_size):
        """发送指令到节点"""
        target_node = self._localizaObjetoCH(target_node_id, self.nodes)
        if not target_node:
            return False
        
        zone_name = target_node["zone"]
        config = self.zone_configs[zone_name]
        selected_modules = self.selected_modules_by_zone.get(zone_name, [])
        rts_cts_flag = 1 if "rts_cts" in selected_modules else 0
        
        path = self._get_path_to_node(target_node)
        if not path:
            return False
        
        current_node = self.base_station
        for next_node in path:
            if next_node != target_node and current_node != self.base_station:
                if rts_cts_flag == 1:
                    current_node["energy"] = self._energy_consumption_tx(
                        current_node["energy"], 
                        self._distance(current_node["x"], current_node["y"], next_node["x"], next_node["y"]),
                        self.rts_cts_size, config
                    )
                    current_node["energy"] = self._energy_consumption_rx(
                        current_node["energy"], 
                        self.rts_cts_size, config
                    )
                
                current_node["energy"] = self._energy_consumption_tx(
                    current_node["energy"], 
                    self._distance(current_node["x"], current_node["y"], next_node["x"], next_node["y"]),
                    command_size, config
                )
                
                if current_node["energy"] <= 0:
                    if "wireless_power" in selected_modules:
                        current_node["energy"] += config["initial_energy"]
                    else:
                        current_node["fault"] = True
                        return False
            
            if current_node != self.base_station and rts_cts_flag == 1:
                next_node["energy"] = self._energy_consumption_tx(
                    next_node["energy"], 
                    self._distance(current_node["x"], current_node["y"], next_node["x"], next_node["y"]),
                    self.rts_cts_size, config
                )
                next_node["energy"] = self._energy_consumption_rx(
                    next_node["energy"], 
                    self.rts_cts_size, config
                )
                next_node["energy"] = self._energy_consumption_rx(
                    next_node["energy"], 
                    command_size, config
                )
            
            if next_node["energy"] <= 0:
                if "wireless_power" in selected_modules:
                    next_node["energy"] += config["initial_energy"]
                else:
                    next_node["fault"] = True
                    return False
            
            current_node = next_node
        
        return True

    def _get_path_to_node(self, target_node):
        """构建从基站到目标节点的反向路径"""
        path = [target_node]
        current = target_node
        
        if not current["routing_information"] or not current["routing_information"][0]:
            return []
        
        while current["routing_information"][0][0] != 0:
            next_hop_id = current["routing_information"][0][0]
            next_node = self._localizaObjetoCH(next_hop_id, self.nodes)
            
            if not next_node:
                return []
            
            path.append(next_node)
            current = next_node
        
        path.reverse()
        return path

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

    loss_data_count, node_fault_list, check_count = sim.run_simulation()