from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json

import logging
from ksim_env.utils.metrics import KMetrics, KFunctionResourceUsage
from ksim_env.utils.benchmark import KBenchmark, cloud_topology
from ksim_env.utils.function_sim import KSimulatorFactory
from ksim_env.utils.simulation import KSimulation
from ksim_env.utils.system import KSystem, AppState, enum_to_str, str_to_enum
from ksim_env.utils.monitor import KMetricsServer, KResourceMonitor
from sim.logging import SimulatedClock
from ksim_env.utils.custom_log import KRuntimeLogger
from sim.core import Environment
import uuid

from typing import Any, Dict

import pandas as pd
logger = logging.getLogger(__name__)

class KsimEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["plot"], "render_fps": 10}
    def __init__(self, config_file: str = None):
        self._parse_config(config_file)
        self._ksim_init()
        
        self.observation_space = spaces.Dict(
            {
                "container_state": spaces.Box(0, 
                                            self.scale_max, 
                                            shape=(self._num_service, len(self.main_states)), dtype=np.int64),
                # 5 là hệ số overcommit
                "invocation_avg": spaces.Box(0, 5*self.num_workers*self.scale_max - 1, shape=(self._num_service, 1), dtype=np.float64), 
            }
        )
        
        self._container_state = np.zeros((self._num_service, len(self.main_states)), dtype=np.int64)
        self._invocation_avg = np.zeros((self._num_service, 1), dtype=np.float64)
        
        # action (x,y,z): chuyển x container từ trạng thái y về trạng thái z
        service_action = [self.scale_max, len(self.main_states), len(self.main_states)] 
        nvec = np.tile(service_action, (len(self.service_profile), 1))
        nvec_flat = nvec.flatten() 
        self.action_space = spaces.MultiDiscrete(nvec_flat)
        
        self._old_invocation_count = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_exec_interval = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_scaler_latency = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_pod_latency = {service_id: 0 for service_id in self.service_profile.keys()}

        self._cluster_cpu_total = 0
        self._cluster_ram_total = 0
        
        for node in self.sim.env.cluster.list_nodes():
            self._cluster_cpu_total += node.capacity.cpu_millis/10
            self._cluster_ram_total += node.capacity.memory/100
            
        self.env_id = str(uuid.uuid4())[:8]  # Short UUID
        logger.info(f"Init environment ID: {self.env_id}")
        self._step_count = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.render_mode = 'plot'

    def _parse_config(self, config_file: str):
        """
        Parses the configuration file to set up the environment.
        This method can be extended to read various parameters from the config file.
        """
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        self.num_servers = self.config.get("num_servers")
        self.main_states = self.config.get("main_states")
        self.scaler_config = self.config.get("scaler_config")
        self.render_dir = self.config.get("render_dir")
        
        # Đẩy timeout ra ngoài để có thể sử dụng timelimit wrapper, môi trường simpy thì cho chạy lặp vô hạn
        self.timeout = self.config.get("timeout")
        self.step_size = self.config.get("step_size")
        self.random_init =  self.config.get("random_init")
        self._max_episode_steps = self.config.get("max_episode_steps", 1440) 
        
        self.service_profile = {}
        for service_id, sv_config in self.config.get("services").items():
            usage_dict = sv_config["state_resource_usage"]
            self.scale_max = sv_config["scale_max"]
            self.scale_min = sv_config["scale_min"]
            self.num_workers = sv_config["num_workers"]
            self.service_profile[service_id] = {
                "image_size": sv_config["image_size"],
                "request_profit": sv_config["request_profit"],
                "num_workers": sv_config["num_workers"],
                "scale_max": sv_config["scale_max"],
                "scale_min": sv_config["scale_min"],
                "trigger_type": sv_config["trigger_type"],
                "avg_execution_time": sv_config["avg_execution_time"],
                "sim_duration": sv_config["sim_duration"],
                "state_resource_usage": {},
                "req_profile_file": self.config.get("req_profile_file"),
                "exec_time_file": self.config.get("exec_time_file")
                
            }
            self.service_profile[service_id]["state_resource_usage"] = {
                str_to_enum[state]: KFunctionResourceUsage(**usage)
                for state, usage in usage_dict.items()
            }
            
        self._num_service = len(self.service_profile)

    def _ksim_init(self):
        self.sim = None
        topology = cloud_topology(self.num_servers)
        
        benchmark = KBenchmark(service_configs=self.service_profile)

        env = Environment()
        env.metrics = KMetrics(env=env, log=KRuntimeLogger(SimulatedClock(env)))
        env.metrics_server = KMetricsServer()
        env.resource_monitor = KResourceMonitor(env, reconcile_interval=1, logging=False)
        if self.random_init:
            for key in self.service_profile.keys():
                self.service_profile[key]["scale_min_org"] = self.service_profile[key]["scale_min"]
                self.service_profile[key]["scale_min"] = np.random.randint(self.service_profile[key]["scale_min_org"], self.service_profile[key]["scale_max"])
        env.simulator_factory = KSimulatorFactory(self.service_profile)
        env.faas = KSystem(env, self.scaler_config)
        
        self.sim = KSimulation(topology=topology, benchmark=benchmark, env=env, name='KSim')
        
    def _get_obs(self):
        i = 0
        faas = self.sim.env.faas
        metrics = self.sim.env.metrics
        for service_id, service_config in self.service_profile.items():
            # Trạng thái null là không tồn tại replica
            self._container_state[i, 0] = service_config["scale_max"] - len(faas.get_replicas2(service_id, AppState.CONCEIVED, lower=False, need_locked=False))
            self._container_state[i, 1] = len(faas.get_replicas(service_id, AppState.UNLOADED_MODEL, need_locked=False))
            self._container_state[i, 2] = len(faas.get_replicas(service_id, AppState.LOADED_MODEL, need_locked=False))
            
            now_invocation = metrics.invocations[service_id]
            self._invocation_avg[i, 0] = (now_invocation - self._old_invocation_count[service_id])
            self._old_invocation_count[service_id] = now_invocation
            i = i + 1
        return {"container_state": self._container_state, "invocation_avg": self._invocation_avg}

    def _get_info(self):
        obs = self._get_obs()
        
        i = 0
        self.info['NULL'] = 0
        self.info['UNLOADED_MODEL'] = 0
        self.info['LOADED_MODEL'] = 0
        self.info['invocation_avg'] = 0
        
        for service_id in self.service_profile.keys():
            self.info['NULL'] += obs["container_state"][i, 0]
            self.info['UNLOADED_MODEL'] += obs["container_state"][i, 1]
            self.info['LOADED_MODEL'] += obs["container_state"][i, 2]
            self.info['invocation_avg'] += obs["invocation_avg"][i, 0]
            i = i + 1
            
        if self.truncated:
            self.info['TimeLimit.truncated'] = True
            self.info['terminal_observation'] = obs
            
        return self.info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Khởi tạo lại môi trường
        self._ksim_init()
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        self._old_invocation_count = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_exec_interval = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_scaler_latency = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_pod_latency = {service_id: 0 for service_id in self.service_profile.keys()}
        
        # Chạy trước 5s để khởi tạo ksim
        self.sim.step(self.step_size)
        observation = self._get_obs()
        info = self._get_info()
        
        logger.info(f"Reset environment ID: {self.env_id}")

        return observation, info
    
    def _get_reward(self):
        # Hướng tới cân bằng request latency và số tài nguyên sử dụng
        now = self.sim.env.now
        reward = 0
        metrics = self.sim.env.metrics
        metrics_server = self.sim.env.metrics_server
        
        for service_id, service_config in self.service_profile.items():
            ram_usage_percent = metrics_server.get_avg_ram_utilization_func(service_id, now-self.step_size, now)/self._cluster_ram_total
            cpu_usage_percent = metrics_server.get_avg_cpu_utilization_func(service_id, now-self.step_size, now)/self._cluster_cpu_total
            
            now_exec_interval = metrics.exec_interval[service_id]
            now_scaler_latency = metrics.scaler_latency[service_id]
            now_pod_latency = metrics.pod_latency[service_id]

            total_exec_interval = now_exec_interval -self._old_exec_interval[service_id] 
            total_scaler_latency = now_scaler_latency -self._old_scaler_latency[service_id] 
            total_pod_latency = now_pod_latency - self._old_pod_latency[service_id] 
            
            # Tỉ số thời gian chờ / thòi gian request thực sự được phục vụ
            
            latency_rate = (0.000001 + total_scaler_latency + total_pod_latency) / (0.000001 + total_exec_interval + total_scaler_latency + total_pod_latency)
            
            reward += 3 - (latency_rate + ram_usage_percent + cpu_usage_percent)
            
            self._old_exec_interval[service_id] = now_exec_interval
            self._old_scaler_latency[service_id] = now_scaler_latency
            self._old_pod_latency[service_id] = now_pod_latency
            
            logger.info(f"Service {service_id} - Reward: {reward}, Latency Rate: {latency_rate}, RAM Usage: {ram_usage_percent}, CPU Usage: {cpu_usage_percent}")

        return reward

    def step(self, action):
        i=0
        env = self.sim.env
        faas = env.faas
        action = np.array(action).reshape((len(self.service_profile), 3))
        self.info["action"] = action
        for service_id in self.service_profile.keys():
            num_replica = action[i, 0]
            from_state = self.main_states[action[i, 1]]
            to_state = self.main_states[action[i, 2]]
            env.process(faas.change_state(service_id, num_replica, from_state, to_state))
            i = i + 1
            
        self.sim.step(self.step_size)
        self.step_count += 1
        
        if self.step_count >= self._max_episode_steps:
            # self.terminated = True
            self.truncated = True
            
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        self.plot_request_details(log_dir=self.render_dir)

    def close(self):
        pass
    
    def plot_request_details(self, log_dir:str=None):
        """
        Plots the latency details for each service.
        """
        import matplotlib.pyplot as plt
        import os
        
        request_details = self.sim.env.metrics.request_details
        
        pod_latency = []
        scaler_latency = []
        exec_interval = []
    
        for service, request in request_details.items():
            for req_id, details in request.items():
                pod_latency.append(details.get('pod_latency', 0))
                scaler_latency.append(details.get('scaler_latency', 0))
                exec_interval.append(details.get('execution_time', 0))
                
        def compute_cdf(data):
            data = np.sort(data)
            cdf = np.arange(1, len(data) + 1) / len(data)
            return data, cdf

        x1, y1 = compute_cdf(pod_latency)
        x2, y2 = compute_cdf(scaler_latency)
        x3, y3 = compute_cdf(exec_interval)

        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

        axs[0].plot(x1, y1, label='Pod Latency', color='tab:blue')
        axs[0].set_title('CDF of Pod Latency')
        axs[0].set_ylabel('CDF')
        axs[0].grid(True)

        axs[1].plot(x2, y2, label='Scaler Latency', color='tab:green')
        axs[1].set_title('CDF of Scaler Latency')
        axs[1].set_ylabel('CDF')
        axs[1].grid(True)

        axs[2].plot(x3, y3, label='Exec Interval', color='tab:red')
        axs[2].set_title('CDF of Exec Interval')
        axs[2].set_xlabel('Seconds')
        axs[2].set_ylabel('CDF')
        axs[2].grid(True)

        fig.tight_layout()

        if log_dir is not None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            fig.savefig(os.path.join(log_dir, f"request_details_{self.env_id}.png"))
        else:
            plt.show()
        plt.close()
        