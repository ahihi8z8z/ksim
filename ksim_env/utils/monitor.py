from dataclasses import dataclass
from typing import Dict, Tuple, List, FrozenSet
from collections import defaultdict
from bisect import bisect_left, bisect_right
import numpy as np

from sim.core import Environment
from sim.faas import FaasSystem
from ksim_env.utils.appstate import AppState 

import logging 

LabelSet = FrozenSet[Tuple[str, str]]  # e.g. frozenset({("function", "fn1"), ("region", "us-east")})
MetricKey = Tuple[str, LabelSet]       # (metric_name, label_set)


@dataclass # order=True thì tự sort theo các field từ trên xuống
class Sample:
    timestamp: float
    value: float
     # sort theo timestamp, dùng cho bisec
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    
@dataclass
class MetricsWindow:
    labels: Dict[str, str]
    metrics: Dict[str, float]
    time: float
    
    def __lt__(self, other):
        return self.time < other.time


class KMetricsServer:
    MAX_SAMPLES = 10000

    def __init__(self):
        self._series: Dict[MetricKey, List[Sample]] = defaultdict(list)
        self._window_index: Dict[str, List[MetricsWindow]] = defaultdict(list)

    def _labels_to_key(self, metric: str, labels: Dict[str, str]) -> MetricKey:
        return (metric, frozenset(labels.items()))

    def put(self, window: MetricsWindow):
        ts = window.time
        for metric_name, value in window.metrics.items():
            key = self._labels_to_key(metric_name, window.labels)
            series = self._series[key]
            
            series.append(Sample(ts, value))

            if len(series) > self.MAX_SAMPLES:
                series.pop(0)
                
        fn = window.labels.get("function")
        if fn:
            self._window_index[fn].append(window)
            if len(self._window_index[fn]) > self.MAX_SAMPLES:
                self._window_index[fn].pop(0)

    def _get_samples(self, metric: str, labels: Dict[str, str], start: float, end: float) -> List[Sample]:
        key = self._labels_to_key(metric, labels)
        series = self._series.get(key, [])
        left = bisect_left(series, Sample(start, 0))
        right = bisect_right(series, Sample(end, float('inf')))
        return series[left:right]
    
    def get_series(self, metric: str, labels: Dict[str, str], start: float, step: float, end: float) -> List[float]:
        samples = self._get_samples(metric, labels, start, end)
        result = []
        timestamps = list(np.arange(start, end + step, step))

        sample_idx = 0
        last_valid_value = None

        for ts in timestamps:
            while sample_idx < len(samples) and samples[sample_idx].timestamp <= ts:
                last_valid_value = samples[sample_idx].value
                sample_idx += 1
            if last_valid_value is not None:
                result.append(last_valid_value) 

        return result

    def get_avg_gauge(self, metric: str, labels: Dict[str, str], start: float, end: float) -> float:
        samples = self._get_samples(metric, labels, start, end)
        values = [s.value for s in samples]
        return float(np.mean(values)) if values else 0.0

    def get_rate_counter(self, metric: str, labels: Dict[str, str], start: float, end: float) -> float:
        samples = self._get_samples(metric, labels, start, end)
        if len(samples) < 2:
            return 0.0
        v_start = samples[0].value
        t_start = samples[0].timestamp
        v_end = samples[-1].value
        t_end = samples[-1].timestamp
        if t_end > t_start and v_end >= v_start:
            return (v_end - v_start) / (t_end - t_start)
        return 0.0

    def get_avg_cpu_utilization(self, labels: Dict[str, str], start: float, end: float) -> float:
        return self.get_avg_gauge("cpu", labels, start, end)

    def get_avg_ram_utilization(self, labels: Dict[str, str], start: float, end: float) -> float:
        return self.get_avg_gauge("memory", labels, start, end)

    # invocation là các request không bị drop
    def get_avg_invocations(self, labels: Dict[str, str], start: float, end: float) -> float:
        return self.get_rate_counter("invocations", labels, start, end)

    def get_avg_drop_count(self, labels: Dict[str, str], start: float, end: float) -> float:
        return self.get_rate_counter("drop_count", labels, start, end)

    def get_avg_request_in(self, labels: Dict[str, str], start: float, end: float) -> float:
        return self.get_rate_counter("request_in", labels, start, end)

    def get_avg_request_interval(self, labels: Dict[str, str], start: float, end: float) -> float:
        rq = self.get_avg_request_in(labels, start, end)
        if rq == 0:
            return 0.0
        interval = self.get_rate_counter("request_interval", labels, start, end)
        return interval / rq

    # latency chỉ tính trên các request không bị drop
    def get_avg_latency(self, labels: Dict[str, str], start: float, end: float) -> float:
        inv = self.get_avg_invocations(labels, start, end)
        if inv == 0:
            return 0.0
        latency = self.get_rate_counter("latency", labels, start, end)
        return latency / inv
    
    def get_avg_all_metrics(self, fn_name: str, window_start: float, window_end: float) -> Dict[str, float]:
        windows = self._window_index.get(fn_name, [])
        if not windows:
            return {
                'cpu': 0.0, 'memory': 0.0, 'latency': 0.0,
                'invocations': 0, 'drop_count': 0,
                'request_in': 0, 'request_interval': 0.0,
                'cold_starts': 0
            }
            
        times = [w.time for w in windows]
        i = bisect_left(times, window_start)
        j = bisect_right(times, window_end)

        matched = windows[i:j]
        if not matched:
            return {
                'cpu': 0.0, 'memory': 0.0, 'latency': 0.0,
                'invocations': 0, 'drop_count': 0,
                'request_in': 0, 'request_interval': 0.0,
                'cold_starts': 0
            }

        start_metrics = matched[0].metrics
        end_metrics = matched[-1].metrics

        cpu_avg = np.mean([w.metrics['cpu'] for w in matched])
        mem_avg = np.mean([w.metrics['memory'] for w in matched])

        # Counter-type metrics
        request_in = end_metrics['request_in'] - start_metrics['request_in']
        invocations = end_metrics['invocations'] - start_metrics['invocations']
        drop_count = end_metrics['drop_count'] - start_metrics['drop_count']
        cold_starts = end_metrics['cold_starts'] - start_metrics['cold_starts']

        # Average latency và request_interval
        latency = (
            (end_metrics['latency'] - start_metrics['latency']) / request_in
            if request_in > 0 else 0.0
        )
        request_interval = (
            (end_metrics['request_interval'] - start_metrics['request_interval']) / request_in
            if request_in > 0 else 0.0
        )

        return {
            'cpu': cpu_avg,
            'memory': mem_avg,
            'latency': latency,
            'invocations': invocations,
            'drop_count': drop_count,
            'request_in': request_in,
            'request_interval': request_interval,
            'cold_starts': cold_starts
        }
    
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
                labels = {"function": function_id}
                metrics = MetricsWindow(
                    labels=labels,
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
                        
                metrics.metrics["latency"] = system_metrics.latency[function_id]
                metrics.metrics["invocations"] = system_metrics.invocations[function_id]
                metrics.metrics["drop_count"] = system_metrics.drop_count[function_id]
                metrics.metrics["request_in"] = system_metrics.request_in[function_id]
                metrics.metrics["request_interval"] = system_metrics.request_interval[function_id]
                metrics.metrics["cold_starts"] = system_metrics.cold_starts[function_id]
                    
                self.metric_server.put(metrics)