import logging
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json

from ksim_env.utils.metrics import KMetrics, KFunctionResourceUsage
from ksim_env.utils.benchmark import KBenchmark, cloud_topology
from ksim_env.utils.function_sim import KSimulatorFactory
from ksim_env.utils.simulation import KSimulation
from ksim_env.utils.system import KSystem, AppState, enum_to_str, str_to_enum
from ksim_env.utils.monitor import KMetricsServer, KResourceMonitor
from ksim_env.utils.custom_log import KRuntimeLogger

from sim.logging import SimulatedClock
from sim.core import Environment
import uuid

from typing import Any, Dict

class KsimEnv_V1(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["plot"], "render_fps": 60}
    def __init__(self, service_config: str = None, system_config: dict = None, render_mode: str = "plot", kwargs: Dict[str, Any] = None):
        self._parse_config(service_config, system_config)
        self._ksim_init()
        
        self.observation_space = spaces.Dict(
            {
                "request_interval": spaces.Box(low=0, high=2.0, shape=(1,), dtype=np.float32),
                "warm_start_invocations": spaces.Box(low=0, high=500.0, shape=(1,), dtype=np.float32),
                "cold_start_invocations": spaces.Box(low=0, high=500.0, shape=(1,), dtype=np.float32),
            }
        )
        
        # action là thay đổi idle threshold
        self.action_space = spaces.Box(low=1, high=100, shape=(1,), dtype=np.float32)
        
        self._avg_ram_utilization = 0
        self._avg_cpu_utilization = 0
        self._avg_power = 0
        self._avg_request_drop = 0 # Cái này là latency trung bình của các request xong trong step
        self._avg_request_interval = 0 # Cái này là interval trung bình của các request đến trong step
        self._request_out_over_step = 0
        self._request_in_over_step = 0
        self._request_drop_over_step = 0
        self._cold_starts_over_step = 0

        self._cluster_cpu_total = 0
        self._cluster_ram_total = 0
        self._power_max = 0
        
        for node in self.sim.env.cluster.list_nodes():
            self._cluster_cpu_total += node.capacity.cpu_millis
            self._cluster_ram_total += node.capacity.memory
            self._power_max += 150
            
        self.env_id = str(uuid.uuid4())[:8]  # Short UUID
        logging.info(f"Init environment ID: {self.env_id}")
        self._step_count = 0
        self.terminated = False
        self.truncated = False
        self.observation = None
        self.info = {}
        
        self.render_mode = render_mode

    def _parse_config(self, service_config: str, system_config: dict):
        """
        Parses the configuration file to set up the environment.
        This method can be extended to read various parameters from the config file.
        """
        with open(service_config, 'r') as f:
            service_config = json.load(f)
            
        self.num_servers = system_config.get("num_servers")
        self.main_states = system_config.get("main_states")
        self.scaler_config = system_config.get("scaler_config")

        self.step_size = system_config.get("step_size")
        self._max_episode_steps = system_config.get("max_episode_steps", 1440) 
        self.server_cap = system_config.get("server_cap")
        self.get_detail = system_config.get("get_detail")
        self.load_balancer = system_config.get("load_balancer", "least_connection").lower()
        self.poll_config = system_config.get("poll_config", {"poll_interval": 0.1, "max_poll_attempts": 40})
        self.random_start = system_config.get("random_start", False)
        
        self.service_profile = {}
        for service_id, sv_config in service_config.items():
            # Hiện tại chỉ có 1 service nên mấy cái này fixed
            self.scale_max = sv_config["scale_max"]
            self.scale_min = sv_config["scale_min"]
            self.num_workers = sv_config["num_workers"]
            self.service_profile[service_id] = sv_config
            self.service_profile[service_id]["random_start"] = self.random_start
            
            usage_dict = sv_config["state_resource_usage"]
            self.service_profile[service_id]["state_resource_usage"] = {
                str_to_enum[state]: KFunctionResourceUsage(**usage)
                for state, usage in usage_dict.items()
            }
            
        self._num_service = len(self.service_profile)

    def _ksim_init(self):
        self.sim = None
        topology = cloud_topology(self.num_servers, self.server_cap)
        
        benchmark = KBenchmark(service_configs=self.service_profile)

        env = Environment()
        
        env.metrics = KMetrics(env=env, log=KRuntimeLogger(SimulatedClock(env)), get_detail=self.get_detail)
        env.metrics_server = KMetricsServer()
        env.resource_monitor = KResourceMonitor(env, reconcile_interval=1, logging=False)
        env.simulator_factory = KSimulatorFactory(self.service_profile)
        env.faas = KSystem(env, self.scaler_config, self.load_balancer, self.poll_config)
        
        self.sim = KSimulation(topology=topology, benchmark=benchmark, env=env, name='KSim')
        
    def _get_obs(self):
        i = 0
        obs = {
            "request_interval": [self._avg_request_interval],
            "warm_start_invocations": [self._request_out_over_step - self._cold_starts_over_step],
            "cold_start_invocations": [self._cold_starts_over_step]
        }
        return obs
    
    def _cal_env_info(self):
        now = self.sim.env.now
        metrics_server = self.sim.env.metrics_server   
        active_nodes = self.sim.env.faas.get_active_nodes()
        
        for service_id in self.service_profile.keys():
            metrics = metrics_server.get_avg_all_metrics(service_id, now-self.step_size, now)
            self._avg_ram_utilization = metrics["memory"] / self._cluster_ram_total
            self._avg_cpu_utilization = metrics["cpu"] / self._cluster_cpu_total
            self._avg_power = (90*len(active_nodes) + 60*self._avg_cpu_utilization) / self._power_max
            
            self._avg_request_drop = metrics["latency"]
            self._avg_request_interval = metrics["request_interval"]
            self._request_drop_over_step = metrics["drop_count"]
            self._cold_starts_over_step = metrics["cold_starts"]
            self._request_out_over_step = metrics["invocations"]
            self._request_in_over_step = metrics["request_in"]
            
    def _get_info(self):  
        if self.get_detail:
            i = 0
            self.info['NULL'] = 0
            self.info['UNLOADED_MODEL'] = 0
            self.info['LOADED_MODEL'] = 0
            faas = self.sim.env.faas
            
            for service_id, service_config in self.service_profile.items():
                self.info['NULL'] = service_config["scale_max"] - len(faas.get_replicas2(service_id, AppState.CONCEIVED, lower=False, need_locked=False))
                self.info['UNLOADED_MODEL'] = len(faas.get_replicas(service_id, AppState.UNLOADED_MODEL, need_locked=False))
                self.info['LOADED_MODEL'] = len(faas.get_replicas(service_id, AppState.LOADED_MODEL, need_locked=False))
                
                self.info['ram_util'] = self._avg_ram_utilization
                self.info['cpu_util'] = self._avg_cpu_utilization
                self.info['power_util'] = self._avg_power
                self.info['request_out_over_step'] = self._request_out_over_step
                self.info['request_in_over_step'] = self._request_in_over_step
                self.info['request_drop_over_step'] = self._request_drop_over_step
                self.info['cold_start_over_step'] = self._cold_starts_over_step
                i = i + 1
            
        if self.truncated:
            self.info['TimeLimit.truncated'] = True
            self.info['terminal_observation'] = self.observation
            self.info['request_details'] = self.sim.env.metrics.request_details
            
        return self.info
    
    def _get_reward(self):
        reward = 0
        for service_id, _ in self.service_profile.items():
            if self._request_out_over_step == 0:
                cold_start_rate = 0
            else:
                # drop cung tinh la cold start
                cold_start_rate = (self._cold_starts_over_step + self._request_drop_over_step) / self._request_in_over_step
                
            reward -= cold_start_rate + self._avg_ram_utilization
            logging.info(f"Service {service_id} - Reward: {reward}, Cold Start Rate: {cold_start_rate}, RAM utilization: {self._avg_ram_utilization}")

        return reward

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Khởi tạo lại môi trường
        self._ksim_init()
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        
        self.sim.step(self.step_size)
        self._cal_env_info()
        self.observation = self._get_obs()
        info = self._get_info()
        
        logging.info(f"Reset environment ID: {self.env_id}")

        return self.observation, info

    def step(self, action):
        i=0
        env = self.sim.env
        faas = env.faas
        idle_threshold = action[0]
        self.info["idle_threshold"] = idle_threshold
        
        for service_id, _ in self.service_profile.items():
            faas.change_idle_threshold(service_id, idle_threshold)
            
        self.sim.step(self.step_size)
        self.step_count += 1
        
        if self.step_count >= self._max_episode_steps:
            # self.terminated = True
            self.truncated = True
            
        self._cal_env_info()
        self. observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()

        return self.observation, reward, self.terminated, self.truncated, info
    
    def render(self):
        return

    def close(self):
        pass
