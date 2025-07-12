from sim.resource import ResourceWindow
from collections import defaultdict, deque

import numpy as np
from sim.core import Environment
from sim.faas import FaasSystem
from ksim_env.utils.appstate import AppState 

import logging
    
logger = logging.getLogger(__name__)    
    
class KMetricsServer:
    MAX_WINDOWS = 1000

    def __init__(self):
        self._windows = defaultdict(self._new_window_deque)

    def _new_window_deque(self):
        return deque(maxlen=self.MAX_WINDOWS)

    def put(self, window: 'ResourceWindow'):
        fn = window.replica.fn_name
        self._windows[fn].append(window)

    def get_avg_resource_utilization_func(self, fn_name: str, resource: str, window_start: float, window_end: float) -> float:
        windows: deque = self._windows.get(fn_name, deque())
        if not windows:
            return 0.0

        matched = [
            w.resources[resource]
            for w in reversed(windows)
            if window_start <= w.time <= window_end
        ]

        return float(np.mean(matched)) if matched else 0.0

    def get_avg_cpu_utilization_func(self, fn_name: str, window_start: float, window_end: float) -> float:
        return self.get_avg_resource_utilization_func(fn_name, 'cpu', window_start, window_end)

    def get_avg_ram_utilization_func(self, fn_name: str, window_start: float, window_end: float) -> float:
        return self.get_avg_resource_utilization_func(fn_name, 'memory', window_start, window_end)
    
class KResourceMonitor:
    def __init__(self, env: Environment, reconcile_interval: int, logging=True):
        self.env = env
        self.reconcile_interval = reconcile_interval
        self.metric_server: KMetricsServer = env.metrics_server
        self.logging = logging

    def run(self):
        faas: FaasSystem = self.env.faas
        while True:
            yield self.env.timeout(self.reconcile_interval)
            now = self.env.now
            for deployment in faas.get_deployments():
                # Lấy các container ở trạng thái 
                replicas = faas.get_replicas2(deployment.name, AppState.CONCEIVED, lower=False, need_locked=False)
                for replica in replicas:
                    utilization = self.env.resource_state.get_resource_utilization(replica)
                    if utilization.is_empty():
                        continue
                    # if self.logging:
                    #     self.env.metrics.log_function_resource_utilization(replica, utilization)
                    self.metric_server.put(ResourceWindow(replica, utilization.list_resources(), now))