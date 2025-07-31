from dataclasses import dataclass
from typing import Dict
from collections import defaultdict, deque

import numpy as np
from sim.core import Environment
from sim.faas import FaasSystem, FunctionReplica
from ksim_env.utils.appstate import AppState 

import logging 
    
@dataclass
class MetricsWindow:
    function: str
    metrics: Dict[str, float]
    time: float

class KMetricsServer:
    MAX_WINDOWS = 1000

    def __init__(self):
        self._windows = defaultdict(self._new_window_deque)

    def _new_window_deque(self):
        return deque(maxlen=self.MAX_WINDOWS)

    def put(self, window: 'MetricsWindow'):
        fn = window.function
        self._windows[fn].append(window)

    def get_avg_gauge(self, fn_name: str, metric: str, window_start: float, window_end: float) -> float:
        windows: deque = self._windows.get(fn_name, deque())
        if not windows:
            return 0.0

        matched = [
            w.metrics[metric]
            for w in reversed(windows)
            if window_start <= w.time <= window_end
        ]

        return float(np.mean(matched)) if matched else 0.0
    
    def get_rate_counter(self, fn_name: str, metric: str, window_start: float, window_end: float) -> float:
        windows = self._windows.get(fn_name, deque())
        matched = [
            (w.time, w.metrics[metric])
            for w in windows
            if window_start <= w.time <= window_end
        ]
        if len(matched) < 2:
            return 0.0

        start_time, start_val = matched[0]
        end_time, end_val = matched[-1]
        return (end_val - start_val) / (end_time - start_time) if end_time > start_time and end_val >= start_val else 0.0
    
    def get_avg_cpu_utilization(self, fn_name: str, window_start: float, window_end: float) -> float:
        return self.get_avg_gauge(fn_name, 'cpu', window_start, window_end)

    def get_avg_ram_utilization(self, fn_name: str, window_start: float, window_end: float) -> float:
        return self.get_avg_gauge(fn_name, 'memory', window_start, window_end)
    
    def get_avg_invocations(self, fn_name: str, window_start: float, window_end: float) -> float:
        return self.get_rate_counter(fn_name, 'invocations', window_start, window_end)
    
    def get_avg_drop_count(self, fn_name: str, window_start: float, window_end: float) -> float:
        return self.get_rate_counter(fn_name, 'drop_count', window_start, window_end)
    
    def get_avg_request_in(self, fn_name: str, window_start: float, window_end: float) -> float:
        return self.get_rate_counter(fn_name, 'request_in', window_start, window_end)
    
    def get_avg_request_interval(self, fn_name: str, window_start: float, window_end: float) -> float:
        invocations = self.get_avg_invocations(fn_name, window_start, window_end)
        if invocations == 0:
            return 0.0
        request_interval = self.get_rate_counter(fn_name, 'request_interval', window_start, window_end)
        return request_interval / invocations

    def get_avg_latency(self, fn_name: str, window_start: float, window_end: float) -> float:
        invocations = self.get_avg_invocations(fn_name, window_start, window_end)
        if invocations == 0:
            return 0.0
        latency = self.get_rate_counter(fn_name, 'latency', window_start, window_end)
        return latency / invocations if latency >= 0 else 0.0
    
    # Reduce repeat logic 
    def get_avg_all_metrics(self, fn_name: str, window_start: float, window_end: float) -> Dict[str, float]:
        windows = self._windows.get(fn_name, deque())
        matched = [ w for w in windows if window_start <= w.time <= window_end ]
        if not matched:
            return { 'cpu': 0.0, 'memory': 0.0, 'latency': 0.0, 'invocations': 0, 'drop_count': 0, 'request_in': 0, 'request_interval': 0 }
        
        start_time, start_metrics = matched[0].time, matched[0].metrics
        end_time, end_metrics = matched[-1].time, matched[-1].metrics
        duration = end_time - start_time if end_time > start_time else 1.0  # Avoid division by zero
        
        ret = {}
        ret['cpu'] = (end_metrics['cpu'] - start_metrics['cpu']) / duration
        ret['memory'] = (end_metrics['memory'] - start_metrics['memory']) / duration
        ret['request_in'] = (end_metrics['request_in'] - start_metrics['request_in']) 
        ret['invocations'] = (end_metrics['invocations'] - start_metrics['invocations']) 
        ret['drop_count'] = (end_metrics['drop_count'] - start_metrics['drop_count']) 
        ret['cold_starts'] = (end_metrics['cold_starts'] - start_metrics['cold_starts'])
        ret['request_interval'] = (end_metrics['request_interval'] - start_metrics['request_interval']) / ret['request_in'] if ret['request_in'] > 0 else 0.0
        ret['latency'] = (end_metrics['latency'] - start_metrics['latency']) / ret['invocations'] if ret['invocations'] > 0 else 0.0
        
        return ret
    
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
            system_metrics = self.env.metrics
            for deployment in faas.get_deployments():
                function_id = deployment.name
                metrics = MetricsWindow(
                    function=function_id,
                    metrics={
                        "cpu": 0.0, 
                        "memory": 0.0,
                        "latency": 0.0, 
                        "invocations": 0, 
                        "cold_starts": 0,
                        "drop_count": 0, 
                        "request_in": 0, 
                        "request_interval": 0},
                    time=now
                )
                # Lấy các container đang hoạt động của deployment
                replicas = faas.get_replicas2(function_id, AppState.CONCEIVED, lower=False, need_locked=False)
                for replica in replicas:
                    utilization = self.env.resource_state.get_resource_utilization(replica)
                    if utilization.is_empty():
                        metrics.metrics["cpu"] += 0.0
                        metrics.metrics["memory"] += 0.0
                    else:
                        metrics.metrics["cpu"] += utilization.get_resource('cpu')
                        metrics.metrics["memory"] +=utilization.get_resource('memory')
                        
                metrics.metrics["latency"] = system_metrics.scaler_latency[function_id]
                metrics.metrics["invocations"] = system_metrics.invocations[function_id]
                metrics.metrics["drop_count"] = system_metrics.drop_count[function_id]
                metrics.metrics["request_in"] = system_metrics.request_in[function_id]
                metrics.metrics["request_interval"] = system_metrics.request_interval[function_id]
                metrics.metrics["cold_starts"] = system_metrics.cold_starts[function_id]
                    
                self.metric_server.put(metrics)