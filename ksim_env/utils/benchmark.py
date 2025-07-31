from sim.topology import Topology
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.faas import FunctionDeployment, Function, FunctionImage, FunctionContainer,  \
    FunctionRequest, KubernetesResourceConfiguration
from sim import docker
from ether.blocks.cells import Cloudlet, BusinessIsp
from ksim_env.utils.traffic_gen import build_rpm, azure_ia_generator
from ksim_env.utils.exectime_gen import build_et_df, azure_et_generator, deterministic_et_generator
from ether.util import parse_size_string
from ether.core import Capacity

import simpy

import pandas as pd
import numpy as np
import logging
from typing import List, Dict


    
def cloud_topology(num_server: int, server_cap) -> Topology:
    t = Topology()
    
    dc = Cloudlet(num_server, 1, backhaul=BusinessIsp())
    cap = Capacity(cpu_millis=server_cap.get("cpu"), memory=parse_size_string(server_cap.get("ram")))
    dc.materialize(t)
    for node in t.get_nodes():
        node.capacity = cap
    
    t.init_docker_registry()
        
    return t

def function_trigger(env: Environment, deployment: FunctionDeployment, et_generator, ia_generator, max_requests=None):
    try:
        if max_requests is None:
            while True:
                ia = next(ia_generator)
                yield env.timeout(ia)
                env.metrics.log_request_in(deployment.fn.name, ia)
                env.process(env.faas.invoke(FunctionRequest(name=deployment.name, size=et_generator())))
        else:
            for _ in range(max_requests):
                ia = next(ia_generator)
                yield env.timeout(ia)
                env.metrics.log_request_in(deployment.fn.name, ia)
                env.process(env.faas.invoke(FunctionRequest(name=deployment.name, size=et_generator())))

    except simpy.Interrupt:
        pass
    except StopIteration:
        logging.error(f'{deployment.name} gen has finished')

class ScalingConfiguration:
    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 1
    scale_zero: bool = False
    
    # average requests per second threshold for scaling
    rps_threshold: int = 20

    # window over which to track the average rps
    alert_window: int = 50  # TODO currently not supported by FaasRequestScaler

    # seconds the rps threshold must be violated to trigger scale up
    rps_threshold_duration: int = 10

    # target average cpu utilization of all replicas, used by HPA
    target_average_utilization: float = 0.5

    # target average rps over all replicas, used by AverageFaasRequestScaler
    target_average_rps: int = 200

    # target of maximum requests in queue
    target_queue_length: int = 75

    target_average_rps_threshold = 0.1
    
    # lstm scaler
    lstm_factor: float = 0.02  # factor to scale the LSTM prediction, default is 0.02
    lstm_model_path: str = None
    scaler_path: str = None

class KBenchmark(Benchmark):
    def __init__(self, service_configs: Dict) -> None:
        self.service_configs = service_configs
        super().__init__()
        
    # Đẩy image lên registry
    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        for service_id, config in self.service_configs.items():
            image_size = parse_size_string(config['image_size'])
            containers.put(docker.ImageProperties(f'{service_id}', image_size, arch='x86'))

        # log all the images in the container
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logging.info('%s, %s, %s', name, tag, images)
                
    def prepare_invocation_profile(self, service_id):
        invocation_profile = self.service_configs[service_id]['invocation_profile']
        if invocation_profile['type'] == 'file':
            random_start = self.service_configs[service_id].get('random_start')
            profile = pd.read_csv(invocation_profile['value'])
            rpm, period, start_day = build_rpm(profile, service_id, random_start)
            ia_generator = azure_ia_generator(rpm, period)
            return ia_generator, start_day 
        else:
            raise ValueError(f"Unsupported invocation profile type: {invocation_profile['type']} for service {service_id}")

    def prepare_execution_time_profile(self, service_id, start_day):
        exec_time_profile = self.service_configs[service_id]['exec_time_profile']
        if exec_time_profile['type'] == 'file':
            profile = pd.read_csv(exec_time_profile['value'])
            et_df = build_et_df(profile, service_id)
            et_generator = azure_et_generator(et_df, start_day)
            return et_generator
        elif exec_time_profile['type'] == 'deterministic':
            value = exec_time_profile['value']
            et_generator = deterministic_et_generator(value)
            return et_generator
        else:
            raise ValueError(f"Unsupported execution time profile type: {exec_time_profile['type']} for service {service_id}")
        
    def run(self, env: Environment):
        # deploy functions
        deployments = self.prepare_deployments()

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        for i in range(len(deployments)):
            service_id = deployments[i].fn.name
            
            ia_generator, start_day = self.prepare_invocation_profile(service_id)
            et_generator = self.prepare_execution_time_profile(service_id, start_day)

            # đẩy request vào hệ thống
            logging.info(f'Start triggering requests of service {service_id}')
            yield from function_trigger(env=env, 
                                        deployment=deployments[i],
                                        et_generator=et_generator,
                                        ia_generator=ia_generator)

    def prepare_deployments(self) -> List[FunctionDeployment]:
        fds = []
        
        for service_id, config in self.service_configs.items():
            image = FunctionImage(image=f'{service_id}')
            func = Function(f'{service_id}', fn_images=[image])
            
            # Run time
            resource_config = KubernetesResourceConfiguration.create_from_str(cpu=config.get("resources")["cpu"], 
                                                                            memory=config.get("resources")["ram"])
            container = FunctionContainer(image, resource_config)
            
            scaling_config = ScalingConfiguration()
            scaling_config.scale_min = config['scale_min']
            scaling_config.scale_max = config['scale_max']
            scaling_config.alert_window = config['alert_window']
            scaling_config.rps_threshold = config['rps_threshold']
            scaling_config.rps_threshold_duration = config['rps_threshold_duration']
            scaling_config.target_average_utilization = config['target_average_utilization']
            scaling_config.target_average_rps = config['target_average_rps']
            scaling_config.target_queue_length = config['target_queue_length']
            scaling_config.target_average_rps_threshold = config['target_average_rps_threshold']
            scaling_config.lstm_factor = config.get('lstm_factor', 0.02)
            scaling_config.lstm_model_path = config.get('lstm_model_path')
            scaling_config.scaler_path = config.get('scaler_path')

            fd = FunctionDeployment(
                func,
                [container],
                scaling_config
            )
            
            fds.append(fd)

        return fds