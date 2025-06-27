from sim.topology import Topology
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.faas import FunctionDeployment, Function, FunctionImage, FunctionContainer, FunctionRequest
from sim import docker
from ether.blocks.cells import Cloudlet, BusinessIsp
from ksim_env.utils.traffic_gen import build_interval_generator, estimate_period

import simpy

import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

sv_config_keys = [
    'image_size',  # kích thước của image, đơn vị là bytes
    'scale_max',  # số lượng replica tối đa của service
    'trigger_type',  # loại trigger của service, có thể là 'http' hoặc 'timer'
    'service_id'
    ]

# Trong azure dataset, Function Invocation Counts được ghi lại theo từng phút.
def azure_ia_generator(profile: pd.DataFrame, HashFunction: str, sim_duration: int):
    data = profile[profile['HashFunction'].str.startswith(HashFunction)]
    row = data.sample(n=1, random_state=42).iloc[0, 4:].values 
    period = 1440
    
    if len(row) < sim_duration:
        sim_duration = len(row)
        
    def randint_step(low, high, step):
        num = (high - low) // step
        return low + np.random.randint(0, num) * step
    
    # chọn lấy 1 chu kì ngẫu nhiên
    start = randint_step(0, len(row) - sim_duration + 1, period) 
    
    # Nhân 10 lên mô phỏng vô hạn
    rpm = np.tile(row[start : start + sim_duration], 10)
    
    # period = estimate_period(rpm)
    ia_gen = build_interval_generator(rpm, period)
    
    t_now = 0.0 # phút
    while t_now < len(rpm):
        arrival_minute = int(t_now)
        if arrival_minute < sim_duration+1:
            rpm[arrival_minute] += 1
        ia =  ia_gen(t_now)
        t_now += ia
        yield ia*60 # trả về theo đơn vị s
        
def cloud_topology(num_server: int) -> Topology:
    t = Topology()
    
    dc = Cloudlet(num_server, 1, backhaul=BusinessIsp())
    dc.materialize(t)
    
    t.init_docker_registry()
    return t

def function_trigger(env: Environment, deployment: FunctionDeployment, f_duration: float, ia_generator, max_requests=None):
    try:
        if max_requests is None:
            while True:
                ia = next(ia_generator)
                yield env.timeout(ia)
                env.metrics.log_request_in(deployment.fn.name)
                env.process(env.faas.invoke(FunctionRequest(name=deployment.name, size=f_duration)))
        else:
            for _ in range(max_requests):
                ia = next(ia_generator)
                yield env.timeout(ia)
                env.metrics.log_request_in(deployment.fn.name)
                env.process(env.faas.invoke(FunctionRequest(name=deployment.name, size=f_duration)))

    except simpy.Interrupt:
        pass
    except StopIteration:
        logging.error(f'{deployment.name} gen has finished')

class ScalingConfiguration:
    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 1
    scale_zero: bool = False

    # window over which to trigger scaler
    alert_window: int = 50  

class KBenchmark(Benchmark):
    def __init__(self, req_profile: pd.DataFrame, service_configs: Dict) -> None:
        self.req_profile = req_profile
        self.service_configs = service_configs
        super().__init__()
        
    # Đẩy image lên registry
    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        for service_id, config in self.service_configs.items():
            image_size = config['image_size']
            containers.put(docker.ImageProperties(f'{service_id}', image_size, arch='x86'))

        # log all the images in the container
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        # deploy functions
        deployments = self.prepare_deployments()

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        for i in range(len(deployments)):
            # logger.info(f'Waiting for deployment {i+1}') 
            # yield env.process(env.faas.poll_available_replica(f'{deployments[i].fn.name}'))

            # tạo request profile
            service_id = deployments[i].fn.name
            trigger_type = self.service_configs[service_id]['trigger_type']
            avg_execution_time = self.service_configs[service_id]['avg_execution_time']
            sim_duration = self.service_configs[service_id]['sim_duration']
            
            # sim_duration tính theo phút nhưng trong config để là giờ cho dễ hiểu
            ia_generator = azure_ia_generator(self.req_profile, HashFunction=service_id, sim_duration=sim_duration*60)

            # đẩy request vào hệ thống
            logger.info(f'Start triggering requests of service {service_id}')
            yield from function_trigger(env=env, 
                                        deployment=deployments[i],
                                        f_duration=avg_execution_time,
                                        ia_generator=ia_generator)

    def prepare_deployments(self) -> List[FunctionDeployment]:
        fds = []
        
        for service_id, config in self.service_configs.items():
            image = FunctionImage(image=f'{service_id}')
            func = Function(f'{service_id}', fn_images=[image])
            
            # Run time
            container = FunctionContainer(image)
            
            scaling_config = ScalingConfiguration()
            scaling_config.scale_min = config['scale_min']
            scaling_config.scale_max = config['scale_max']

            fd = FunctionDeployment(
                func,
                [container],
                scaling_config
            )
            
            fds.append(fd)

        return fds