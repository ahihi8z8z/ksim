import logging
import time

from skippy.core.scheduler import Scheduler

from sim.benchmark import Benchmark
from sim.core import Environment, timeout_listener
from sim.docker import ContainerRegistry

from sim.resource import ResourceState, ResourceMonitor
from sim.skippy import SimulationClusterContext
from sim.topology import Topology

import json


logger = logging.getLogger(__name__)

class KSimulation:
    def __init__(self, topology: Topology, benchmark: Benchmark, env: Environment = None, timeout=None, name=None):
        self.env = env or Environment()
        self.topology = topology
        self.benchmark = benchmark
        self.timeout = timeout
        self.name = name
        
        logger.info('initializing simulation, benchmark: %s, topology nodes: %d',
                    type(self.benchmark).__name__, len(self.topology.nodes))

        env = self.env

        env.benchmark = self.benchmark
        env.topology = self.topology

        self.init_environment(env)

        then = time.time()

        if self.timeout:
            logger.info('starting timeout listener with timeout %d', self.timeout)
            env.process(timeout_listener(env, then, self.timeout))

        logger.info('starting resource monitor')
        env.process(env.resource_monitor.run())

        logger.info('setting up benchmark')
        self.benchmark.setup(env)

        logger.info('starting faas system')
        env.faas.start()
        
        logger.info('starting energy monitor system')
        env.energy_monitor = EnergyMonitor(env, env.power_config)

        env.process(env.energy_monitor.run())

        logger.info('starting benchmark process')
        env.process(self.benchmark.run(env))

        logger.info('executing simulation')

    def init_environment(self, env):
        # This is ksim_env specific initialization
        if not env.simulator_factory:
            raise ValueError('ksimulator_factory is not set in the environment')
        
        if not env.faas:
            raise ValueError('ksystem is not set in the environment')
        
        if not env.metrics:
            raise ValueError('kmetrics is not set in the environment')
        
        if not env.metrics_server:
            raise ValueError('kmetrics_server is not set in the environment')
        
        if not env.resource_monitor:
            raise ValueError('kresource_monitor is not set in the environment')
        
        if not env.container_registry:
            env.container_registry = self.create_container_registry()
            
        # Others is default
        if not env.cluster:
            env.cluster = SimulationClusterContext(env)

        if not env.scheduler:
            env.scheduler = self.create_scheduler(env)
        
        if not env.resource_state:
            env.resource_state = ResourceState()

    def create_container_registry(self):
        return ContainerRegistry()

    def create_scheduler(self, env):
        return Scheduler(env.cluster)

    def step(self, interval: int = 1):
        """
        Run the simulation for a specified interval.
        """
        logger.info('now: %s, running simulation for %d seconds', self.env.now, interval)
        self.env.run(until=self.env.now + interval)
        
class EnergyMonitor:
    def __init__(self, env: Environment, power_config, reconile_interval: int = 1):
        self.env = env
        self.reconile_interval = reconile_interval
        self.power_config = power_config
        
        self.power_list = []
        self.avg_power = 0
        
    def run(self):
        faas = self.env.faas
        env = self.env
        
        while True:
            current_energy = 0

            for nodename in env.node_states:

                if not env.resource_state.node_resource_utilizations or not env.resource_state.get_node_resource_utilization(nodename).total_utilization.get_resource('cpu'):
                    continue

                if env.resource_state.node_resource_utilizations:
                    capacity = env.node_states[nodename].capacity.cpu_millis
                    current_using = env.resource_state.get_node_resource_utilization(nodename).total_utilization.get_resource('cpu')
                    
                    base_power = self.power_config['base'] * self.power_config['max_power']
                    usage = (self.power_config['max_power'] - base_power) * (current_using / capacity)

                    current_energy += (base_power + usage)

                    # logger.critical(f'Monitor E {capacity} | {current_using} | {current_energy}')

            
            self.power_list.append(current_energy)

            if len(self.power_list) == 0:
                continue

            if len(self.power_list) == 1:
                self.avg_power = current_energy / 2

            if len(self.power_list) == 2:
                self.avg_power = (sum(self.power_list) / 2)

            if len(self.power_list) > 2:
                self.power_list.pop(0)
                self.avg_power = (sum(self.power_list) / 2)

            yield self.env.timeout(self.reconile_interval)