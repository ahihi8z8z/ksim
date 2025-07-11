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

logger = logging.getLogger(__name__)

class KsimEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": ["plot"], "render_fps": 60}
    def __init__(self, render_dir: str, config_file: str = None):
        self._parse_config(config_file)
        self._ksim_init()
        
        self.observation_space = spaces.Dict(
            {
                "container_state": spaces.Box(0, 
                                            self.scale_max, 
                                            shape=(self._num_service, len(self.main_states)), dtype=np.int64),
                # 5 là hệ số overcommit
                "request_rate": spaces.Box(0, 5*self.num_workers*self.scale_max - 1, shape=(self._num_service, 1), dtype=np.float64), 
            }
        )
        
        self._container_state = np.zeros((self._num_service, len(self.main_states)), dtype=np.int64)
        self._request_rate = np.zeros((self._num_service, 1), dtype=np.float64)
        
        # action (x,y,z): chuyển x container từ trạng thái y về trạng thái z
        service_action = [self.scale_max, len(self.main_states), len(self.main_states)] 
        nvec = np.tile(service_action, (len(self.service_profile), 1))
        nvec_flat = nvec.flatten() 
        self.action_space = spaces.MultiDiscrete(nvec_flat)
        
        self._old_request_in = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_request_drop = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_exec_interval = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_scaler_latency = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_pod_latency = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_request_out = {service_id: 0 for service_id in self.service_profile.keys()}
        
        self._ram_usage_percent = 0
        self._cpu_usage_percent = 0
        self._power_avg = 0
        self._exec_interval_over_step = 0
        self._scaler_latency_over_step = 0
        self._pod_latency_over_step = 0
        self._request_out_over_step = 0
        self._request_in_over_step = 0
        self._request_drop_over_step = 0

        self._cluster_cpu_total = 0
        self._cluster_ram_total = 0
        self._power_max = 0
        
        for node in self.sim.env.cluster.list_nodes():
            self._cluster_cpu_total += node.capacity.cpu_millis
            self._cluster_ram_total += node.capacity.memory
            self._power_max += 150
            
        self.env_id = str(uuid.uuid4())[:8]  # Short UUID
        logger.info(f"Init environment ID: {self.env_id}")
        self._step_count = 0
        self.terminated = False
        self.truncated = False
        self.observation = None
        self.info = {}
        
        self.render_mode = 'plot'
        self.render_dir = render_dir

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

        self.timeout = self.config.get("timeout")
        self.step_size = self.config.get("step_size")
        self.random_init =  self.config.get("random_init")
        self._max_episode_steps = self.config.get("max_episode_steps", 1440) 
        self.server_cap = self.config.get("server_cap")
        
        self.service_profile = {}
        for service_id, sv_config in self.config.get("services").items():
            # Hiện tại chỉ có 1 service nên mấy cái này fix cứng
            self.scale_max = sv_config["scale_max"]
            self.scale_min = sv_config["scale_min"]
            self.num_workers = sv_config["num_workers"]
            self.service_profile[service_id] = sv_config
            
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
            
            self._request_rate[i, 0] = self._request_in_over_step/self.step_size

            i = i + 1
        return {"container_state": self._container_state, "request_rate": self._request_rate}

    def _cal_env_info(self):
        now = self.sim.env.now
        metrics = self.sim.env.metrics
        metrics_server = self.sim.env.metrics_server   
        active_nodes = self.sim.env.faas.get_active_nodes()
        
        for service_id in self.service_profile.keys():
            self._ram_usage_percent = metrics_server.get_avg_ram_utilization_func(service_id, now-self.step_size, now)/self._cluster_ram_total
            self._cpu_usage_percent = metrics_server.get_avg_cpu_utilization_func(service_id, now-self.step_size, now)/self._cluster_cpu_total
            self._power_avg = (90*len(active_nodes) + 60*self._cpu_usage_percent) / self._power_max
            
            now_request_drop = metrics.drop_count[service_id]
            now_request_out = metrics.request_out[service_id]
            now_request_in = metrics.request_in[service_id]
            now_exec_interval = metrics.exec_interval[service_id]
            now_scaler_latency = metrics.scaler_latency[service_id]
            now_pod_latency = metrics.pod_latency[service_id]
            
            self._exec_interval_over_step = now_exec_interval - self._old_exec_interval[service_id] 
            self._scaler_latency_over_step = now_scaler_latency - self._old_scaler_latency[service_id] 
            self._pod_latency_over_step = now_pod_latency - self._old_pod_latency[service_id] 
            self._request_drop_over_step = now_request_drop - self._old_request_drop[service_id]
            self._request_out_over_step = now_request_out - self._old_request_out[service_id]
            self._request_in_over_step = now_request_in - self._old_request_in[service_id]
            
            self._old_exec_interval[service_id] = now_exec_interval
            self._old_scaler_latency[service_id] = now_scaler_latency
            self._old_pod_latency[service_id] = now_pod_latency
            self._old_request_drop[service_id] = now_request_drop
            self._old_request_out[service_id] = now_request_out
            self._old_request_in[service_id] = now_request_in
            
    def _get_info(self):  
        i = 0
        self.info['NULL'] = 0
        self.info['UNLOADED_MODEL'] = 0
        self.info['LOADED_MODEL'] = 0
        
        for _ in self.service_profile.keys():
            self.info['NULL'] = self.observation["container_state"][i, 0]
            self.info['UNLOADED_MODEL'] = self.observation["container_state"][i, 1]
            self.info['LOADED_MODEL'] = self.observation["container_state"][i, 2]
            
            self.info['ram_util'] = self._ram_usage_percent
            self.info['cpu_util'] = self._cpu_usage_percent
            self.info['power_util'] = self._power_avg
            self.info['request_out_over_step'] = self._request_out_over_step
            self.info['request_in_over_step'] = self._request_in_over_step
            self.info['request_drop_over_step'] = self._request_drop_over_step
            self.info['request_details'] = self.sim.env.metrics.request_details
            i = i + 1
            
        if self.truncated:
            self.info['TimeLimit.truncated'] = True
            self.info['terminal_observation'] = self.observation
            
        return self.info
    
    def _get_reward(self):
        reward = 0
        # Hướng tới cân bằng request latency, tỉ lệ drop và tài nguyên sử dụng
        for service_id, _ in self.service_profile.items():
            if self._request_in_over_step == 0:
                blocking_rate = 0
            else:
                blocking_rate = self._request_drop_over_step / self._request_in_over_step
            
            if self._request_out_over_step == 0:
                latency = 0
            else:
                # 4 là cold start time
                latency = (self._pod_latency_over_step + self._scaler_latency_over_step)/(4*self._request_out_over_step)
                
            reward += 3 - (blocking_rate + latency + self._power_avg)
            
            logger.info(f"Service {service_id} - Reward: {reward}, Blocking Rate: {blocking_rate}, Latency/Rq: {latency}, Power: {self._power_avg}")

        return reward

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Khởi tạo lại môi trường
        self._ksim_init()
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        self._old_request_in = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_request_drop = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_exec_interval = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_scaler_latency = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_pod_latency = {service_id: 0 for service_id in self.service_profile.keys()}
        self._old_request_out = {service_id: 0 for service_id in self.service_profile.keys()}
        
        self.sim.step(self.step_size)
        self._cal_env_info()
        self.observation = self._get_obs()
        info = self._get_info()
        
        logger.info(f"Reset environment ID: {self.env_id}")

        return self.observation, info

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
            
        self._cal_env_info()
        self. observation = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()

        return self.observation, reward, self.terminated, self.truncated, info
    
    def render(self):
        return
        # if self.step_count % self.metadata["render_fps"] == 0:
        #     self.plot_request_details(log_dir=self.render_dir)

    def close(self):
        pass
        
