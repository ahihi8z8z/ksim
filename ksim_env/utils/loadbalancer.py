import logging
from collections import defaultdict
from typing import Dict, List

from sim.core import Environment
from sim.faas import FunctionRequest
from ksim_env.utils.replica import KFunctionReplica
from ksim_env.utils.appstate import AppState



class KRoundRobinLoadBalancer:
    env: Environment
    replicas: Dict[str, List[KFunctionReplica]]

    def __init__(self, env, replicas) -> None:
        super().__init__()
        self.env = env
        self.replicas = replicas
        self.counters = defaultdict(lambda: 0)

    def get_running_replicas(self, function: str):
        return [replica for replica in self.replicas[function] if replica.state.value >= AppState.LOADED_MODEL.value]

    def next_replica(self, request: FunctionRequest) -> KFunctionReplica:
        replicas = self.get_running_replicas(request.name)
        i = self.counters[request.name] % len(replicas)
        self.counters[request.name] = (i + 1) % len(replicas)

        replica = replicas[i]

        return replica
    
class LeastConnectionLoadBalancer:
    env: Environment
    replicas: Dict[str, List[KFunctionReplica]]

    def __init__(self, env, replicas) -> None:
        super().__init__()
        self.env = env
        self.replicas = replicas

    def get_running_replicas(self, function: str):
        return [replica for replica in self.replicas[function] if replica.state.value >= AppState.LOADED_MODEL.value]

    def next_replica(self, request: FunctionRequest) -> KFunctionReplica:
        replicas = self.get_running_replicas(request.name)
        choosen = replicas[0]
        min_queue = len(choosen.simulator.queue.queue)
        for replica in replicas:
            queue_len = len(replica.simulator.queue.queue)
            if queue_len < min_queue:
                choosen = replica
                min_queue = queue_len

        return choosen