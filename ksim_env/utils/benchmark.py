from sim.topology import Topology
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.faas import FunctionDeployment, Function, FunctionImage, FunctionContainer, FunctionRequest
from sim import docker
from ether.blocks.cells import Cloudlet, BusinessIsp
from ksim_env.utils.traffic_gen import build_rpm, azure_ia_generator
from ksim_env.utils.exectime_gen import build_et_df, azure_et_generator

import simpy

import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
        
def cloud_topology(num_server: int) -> Topology:
    t = Topology()
    
    dc = Cloudlet(num_server, 1, backhaul=BusinessIsp())
    dc.materialize(t)
    
    t.init_docker_registry()
    return t

def function_trigger(env: Environment, deployment: FunctionDeployment, et_generator, ia_generator, max_requests=None):
    try:
        if max_requests is None:
            while True:
                ia = next(ia_generator)
                yield env.timeout(ia)
                env.metrics.log_request_in(deployment.fn.name)
                env.process(env.faas.invoke(FunctionRequest(name=deployment.name, size=et_generator())))
        else:
            for _ in range(max_requests):
                ia = next(ia_generator)
                yield env.timeout(ia)
                env.metrics.log_request_in(deployment.fn.name)
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

    # window over which to trigger scaler
    alert_window: int = 50  

class KBenchmark(Benchmark):
    def __init__(self, service_configs: Dict) -> None:
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
            sim_duration = self.service_configs[service_id]['sim_duration']
            req_profile = pd.read_csv(self.service_configs[service_id]['req_profile_file'])
            exectime_profile = pd.read_csv(self.service_configs[service_id]['exec_time_file'])
            
            # sim_duration tính theo phút nhưng trong config để là giờ cho dễ hiểu
            rpm, period, start_day = build_rpm(req_profile, HashFunction=service_id, sim_duration=sim_duration*60)
            ia_generator = azure_ia_generator(rpm, period)
            
            et_df = build_et_df(exectime_profile, service_id)
            et_generator = azure_et_generator(et_df, start_day)

            # đẩy request vào hệ thống
            logger.info(f'Start triggering requests of service {service_id}')
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