from sim.metrics import Metrics
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class KFunctionResourceUsage:
    time: float = 0.0 # in seconds. time bằng 0 có nghĩa là thời gian tồn tại trạng thái không xác định.
    ram: float = 0.0 # Có, in bytes
    cpu: float = 0.0 # Có, in % of total host CPU
    gpu: float = 0.0 #
    disk: float = 0.0 # in bps
    network: float = 0.0 # in bps

    def __init__(self, cpu: float, time: float, gpu: float, network: float, ram: float, disk: float):
        self.cpu = cpu
        self.time = time
        self.gpu = gpu
        self.network = network
        self.ram = ram
        self.disk = disk

    def __len__(self):
        return 6

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

class KMetrics(Metrics):
    """
    KMetrics extends the base Metrics class to provide additional functionality.
    """

    def __init__(self, env, log) -> None:
        super().__init__(env, log)
        self.scaler_latency = defaultdict(float)
        self.pod_latency = defaultdict(float)
        self.exec_interval = defaultdict(float)
        self.drop_count = defaultdict(int)

    def log_load(self, function_name, replicas):
        self.log('load model', replicas, function_name=function_name)
        
    def log_unload(self, function_name, replicas):
        self.log('unload model', replicas, function_name=function_name)
        
    def log_invocation(self, function_name, function_image, node_name, t_wait, t_start, t_exec, replica_id, **kwargs):
        function = self.env.faas.get_function_index()[function_image]
        mem = function.get_resource_requirements().get('memory')
        
        self.scaler_latency[function_name] +=  t_wait

        self.log('invocations', {'t_wait': t_wait, 't_exec': t_exec, 't_start': t_start, 'memory': mem, **kwargs},
                 function_name=function_name,
                 function_image=function_image, node=node_name, replica_id=replica_id)
        
    def log_fet(self, function_name, function_image, node_name, t_fet_start, t_fet_end, replica_id, request_id,
                **kwargs):
        self.exec_interval[function_name] += (t_fet_end - t_fet_start)
        self.pod_latency[function_name] += (kwargs.get('t_wait_end') - kwargs.get('t_wait_start'))
        
        self.log('fets', {'t_fet_start': t_fet_start, 't_fet_end': t_fet_end, **kwargs},
                 function_name=function_name,
                 function_image=function_image, node=node_name, replica_id=replica_id, request_id=request_id)
        
    def log_drop(self, function_name, request_id):
        self.drop_count[function_name] += 1
        self.log('drops', {'request_id': request_id}, function_name=function_name)