import logging
import time
from typing import List, Tuple

from skippy.core.scheduler import Scheduler
from skippy.core.predicates import Predicate, PodFitsResourcesPred, CheckNodeLabelPresencePred
from skippy.core.priorities import Priority, BalancedResourcePriority, LatencyAwareImageLocalityPriority, CapabilityPriority, DataLocalityPriority, LocalityTypePriority

from sim.benchmark import Benchmark
from sim.core import Environment, timeout_listener
from sim.docker import ContainerRegistry
from sim.resource import ResourceState
from sim.skippy import SimulationClusterContext
from sim.topology import Topology

from ksim_env.utils.scheduler import MostRequestedPriority



class KSimulation:
    def __init__(self, topology: Topology, benchmark: Benchmark, env: Environment = None, timeout=None, name=None):
        self.env = env or Environment()
        self.topology = topology
        self.benchmark = benchmark
        self.timeout = timeout
        self.name = name
        
        logging.info('initializing simulation, benchmark: %s, topology nodes: %d',
                    type(self.benchmark).__name__, len(self.topology.nodes))

        env = self.env

        env.benchmark = self.benchmark
        env.topology = self.topology

        self.init_environment(env)

        then = time.time()

        if self.timeout:
            logging.info('starting timeout listener with timeout %d', self.timeout)
            env.process(timeout_listener(env, then, self.timeout))

        logging.info('starting resource monitor')
        env.process(env.resource_monitor.run())

        logging.info('setting up benchmark')
        self.benchmark.setup(env)

        logging.info('starting faas system')
        env.faas.start()

        logging.info('starting benchmark process')
        env.process(self.benchmark.run(env))

        logging.info('executing simulation')

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
        default_predicates: List[Predicate] = [
            PodFitsResourcesPred(),
            CheckNodeLabelPresencePred(['data.skippy.io/storage'], False) 
        ]

        ## Ưu tiên xếp vào node nhiều tải để tiết kiệm power
        default_priorities: List[Tuple[float, Priority]] = [
            (10.0, MostRequestedPriority()),
            (1.0, BalancedResourcePriority()),
            (1.0, CapabilityPriority()),
        ]
        return Scheduler(
            cluster_context=env.cluster, 
            percentage_of_nodes_to_score=100, 
            predicates=default_predicates,
            priorities=default_priorities,
        )

    def step(self, interval: int = 1):
        """
        Run the simulation for a specified interval.
        """
        logging.debug('now: %s, running simulation for %d seconds', self.env.now, interval)
        self.env.run(until=self.env.now + interval)