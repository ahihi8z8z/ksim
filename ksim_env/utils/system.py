import logging
import simpy
from collections import defaultdict, Counter
from typing import Dict, List

from sim.core import Environment
from sim.faas import FunctionDeployment, FunctionContainer, FunctionRequest, FunctionSimulator
from sim.skippy import create_function_pod
from sim.faas.core import FaasSystem

from ksim_env.utils.scaler import FaasRequestScaler, AverageFaasRequestScaler, AverageQueueFaasRequestScaler, HorizontalPodAutoscaler
from ksim_env.utils.replica import KFunctionReplica
from ksim_env.utils.appstate import AppState

logger = logging.getLogger(__name__)

# This is the application state machine for the KSystem.
# Mỗi trạng thái tiêu thụ một số lượng tài nguyên nhất định.
# NULL, UNLOADED_MODEL, LOADED_MODEL là các trạng thái có thể điều khiển được.
# Còn lại là các trạng thái trung gian.
    
str_to_enum = {s.name: s.value for s in AppState}

enum_to_str = {s.value: s.name for s in AppState}
    
class KSystem(FaasSystem):
    def __init__(self, env: Environment, scaler_config: Dict) -> None:
        self.env = env
        self.function_containers = dict()
        # collects all KFunctionReplicas under the name of the corresponding FunctionDeployment
        self.replicas = defaultdict(list)

        self.request_queue = simpy.Store(env)
        self.scheduler_queue = simpy.Store(env)

        # TODO let users inject LoadBalancer
        self.load_balancer = KRoundRobinLoadBalancer(env, self.replicas)

        self.functions_deployments: Dict[str, FunctionDeployment] = dict()
        self.replica_count: Dict[str, int] = dict()
        self.functions_definitions = Counter()
        
        self.scaler_config = scaler_config
        self.faas_scalers: Dict[str, FaasRequestScaler] = dict()
        self.avg_faas_scalers: Dict[str, AverageFaasRequestScaler] = dict()
        self.queue_faas_scalers: Dict[str, AverageQueueFaasRequestScaler] = dict()
        
        self.states = ["NULL", "UNLOADED_MODEL", "LOADED_MODEL"]
        self.forward_transitions = [self.scale_up, self.do_load_model]
        self.backward_transitions = [self.scale_down, self.do_unload_model]
        
    def get_deployments(self) -> List[FunctionDeployment]:
        return list(self.functions_deployments.values())

    def get_function_index(self) -> Dict[str, FunctionContainer]:
        return self.function_containers
    
    def discover(self, function: str) -> List[KFunctionReplica]:
        return [replica for replica in self.replicas[function] if replica.state == AppState.UNLOADED_MODEL]
    
    def suspend(self, function_name: str):
        if function_name not in self.functions_deployments.keys():
            raise ValueError

        # TODO interrupt startup of function containers that are starting
        replicas: List[KFunctionReplica] = self.discover(function_name)
        self.scale_down(function_name, len(replicas))

        self.env.metrics.log_function_deployment_lifecycle(self.functions_deployments[function_name], 'suspend')

    def get_replicas(self, fn_name: str, state=None, need_locked: bool = True) -> List[KFunctionReplica]:
        if state is None:
            return self.replicas[fn_name]
        
        if need_locked:
            return [replica for replica in self.replicas[fn_name] if replica.state.value == state.value and replica.locked is False] 
        

        return [replica for replica in self.replicas[fn_name] if replica.state.value == state.value]
    
    # Lấy những replica có trạng thái cao/thấp hơn
    def get_replicas2(self, fn_name: str, state=None, lower=True, need_locked=True) -> List[KFunctionReplica]:
        if state is None:
            return self.replicas[fn_name]
        if lower:
            if need_locked:
                return [replica for replica in self.replicas[fn_name] if replica.state.value <= state.value and replica.locked is False] 
            return [replica for replica in self.replicas[fn_name] if replica.state.value <= state.value]
        else:
            if need_locked:
                return [replica for replica in self.replicas[fn_name] if replica.state.value >= state.value and replica.locked is False]
            return [replica for replica in self.replicas[fn_name] if replica.state.value >= state.value]
    
    def deploy(self, fd: FunctionDeployment):
        if fd.name in self.functions_deployments:
            raise ValueError('function already deployed')

        self.functions_deployments[fd.name] = fd
        
        self.faas_scalers[fd.name] = FaasRequestScaler(fd, self.env)
        self.avg_faas_scalers[fd.name] = AverageFaasRequestScaler(fd, self.env)
        self.queue_faas_scalers[fd.name] = AverageQueueFaasRequestScaler(fd, self.env)

        if self.scaler_config.get("scale_by_requests"):
            self.env.process(self.faas_scalers[fd.name].run())
        if self.scaler_config.get("scale_by_average_requests_per_replica"):
            self.env.process(self.avg_faas_scalers[fd.name].run())
        if self.scaler_config.get("scale_by_queue_requests_per_replica"):
            self.env.process(self.queue_faas_scalers[fd.name].run())

        for f in fd.fn_containers:
            self.function_containers[f.image] = f

        # TODO log metadata
        self.env.metrics.log_function_deployment(fd)
        self.env.metrics.log_function_deployment_lifecycle(fd, 'deploy')
        logger.info('deploying function %s with scale_min=%d', fd.name, fd.scaling_config.scale_min)
        yield from self.change_state(fd.name, fd.scaling_config.scale_min, "NULL", "LOADED_MODEL")  
        

    def deploy_replica(self, fd: FunctionDeployment, fn: FunctionContainer, services: List[FunctionContainer]):
        """
        Creates and deploys a KFunctionReplica for the given FunctionContainer.
        In case no node supports the given FunctionContainer, the services list dictates which FunctionContainer to try next.
        In case no FunctionContainer can be hosted, the scheduling process terminates and logs the failed attempt
        """
        replica = self.create_replica(fd, fn)
        self.replicas[fd.name].append(replica)
        self.env.metrics.log_queue_schedule(replica)
        self.env.metrics.log_function_replica(replica)
        yield self.scheduler_queue.put((replica, services))

    def invoke(self, request: FunctionRequest):
        if request.name not in self.functions_deployments.keys():
            logger.warning('invoking non-existing function %s', request.name)
            return

        t_received = self.env.now

        replicas = self.get_replicas2(request.name, AppState.LOADED_MODEL, False)
        if not replicas:
            retry = 0
            while (len(self.get_replicas2(request.name, AppState.LOADED_MODEL, False)) < 1)  and (retry < 74):
                yield self.env.timeout(0.5)
                retry += 1
            
        replicas = self.get_replicas2(request.name, AppState.LOADED_MODEL, False)
        # Thử drop request sau 30s xem
        if not replicas:
            self.env.metrics.log_drop(request.name, request.request_id)
            return
        
        logger.debug('asking load balancer for replica for request %d', request.request_id)
        replica = self.next_replica(request)

        logger.debug('dispatching request %s:%d to %s', request.name, request.request_id, replica.node.name)

        t_start = self.env.now
        
        self.env.metrics.log_start_exec(request, replica)
        yield from replica.simulator.invoke(self.env, replica, request)
        self.env.metrics.log_stop_exec(request, replica)

        t_end = self.env.now

        t_wait = t_start - t_received
        t_exec = t_end - t_start
        self.env.metrics.log_invocation(request, replica.image, replica.node.name, t_wait, t_start,
                                        t_exec, id(replica))

    def remove(self, fn: FunctionDeployment):
        self.env.metrics.log_function_deployment_lifecycle(fn, 'remove')

        replica_count = self.replica_count[fn.name]
        yield from self.scale_down(fn.name, replica_count)
        
        self.faas_scalers[fn.name].stop()
        self.avg_faas_scalers[fn.name].stop()
        self.queue_faas_scalers[fn.name].stop()

        del self.functions_deployments[fn.name]
        del self.faas_scalers[fn.name]
        del self.avg_faas_scalers[fn.name]
        del self.queue_faas_scalers[fn.name]
        del self.replica_count[fn.name]
        for container in fn.fn_containers:
            del self.functions_definitions[container.image]

    def scale_down(self, fn_name: str, remove: int):
        env = self.env
        can_remove = int(remove)
        if can_remove <= 0:
            return
        
        running_replicas = len(self.get_replicas(fn_name, AppState.UNLOADED_MODEL))
        if running_replicas < remove:
            yield from self.poll_available_replica(fn=fn_name, num_entry=can_remove, interval=0.5, 
                                                    state=AppState.UNLOADED_MODEL, max_retry=60)
            
        running_replicas = len(self.get_replicas(fn_name, AppState.UNLOADED_MODEL))

        if running_replicas < remove:
            can_remove = running_replicas

        scale_min = self.functions_deployments[fn_name].scaling_config.scale_min
        if self.replica_count.get(fn_name, 0) - can_remove < scale_min:
            can_remove = self.replica_count.get(fn_name, 0) - scale_min

        if can_remove <= 0:
            logger.debug('Function %s wanted to scale down, but no replicas were removed', fn_name)
            return

        logger.info(f'scaling down {fn_name} by {can_remove}')
        
        replicas = self.choose_replicas_to_remove(fn_name, can_remove)
        self.env.metrics.log_scaling(fn_name, -can_remove)
        
        for replica in replicas:
            replica.locked = True
        
        for replica in replicas:
            node = replica.node.skippy_node

            env.metrics.log_teardown(replica)
            yield from replica.simulator.teardown(env, replica)

            self.env.cluster.remove_pod_from_node(replica.pod, node)


            env.metrics.log('allocation', {
                'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
                'mem': 1 - (node.allocatable.memory / node.capacity.memory)
            }, node=node.name)
            self.replica_count[replica.fn_name] -= 1
            self.functions_definitions[replica.image] -= 1

    def choose_replicas_to_remove(self, fn_name: str, n: int):
        # TODO: không scale down các replica đang có request
        running_replicas = self.get_replicas(fn_name, AppState.UNLOADED_MODEL)
        return running_replicas[len(running_replicas) - n:]


    def scale_up(self, fn_name: str, replicas: int):
        logger.debug('request to scale up function %s by %d replicas', fn_name, replicas)
        fd = self.functions_deployments[fn_name]
        config = fd.scaling_config
        ranking = fd.ranking

        scale = int(replicas)
        if self.replica_count.get(fn_name, None) is None:
            self.replica_count[fn_name] = 0

        if self.replica_count[fn_name] >= config.scale_max:
            logger.debug('Function %s wanted to scale up, but maximum number of replicas reached', fn_name)
            return

        # check whether request would exceed maximum number of containers for the function and reduce to scale to max
        if self.replica_count[fn_name] + replicas > config.scale_max:
            reduce = self.replica_count[fn_name] + replicas - config.scale_max
            scale = replicas - reduce

        if scale == 0:
            logger.debug('Function %s wanted to scale up, but no new replicas were requested', fn_name)
            return
        
        actually_scaled = 0
        for index, service in enumerate(fd.get_services()):
            # check whether service has capacity, otherwise continue
            leftover_scale = scale
            max_replicas = int(ranking.function_factor[service.image] * config.scale_max)

            # check if scaling all new pods would exceed the maximum number of replicas for this function container
            if max_replicas * config.scale_max < leftover_scale + self.functions_definitions[
                service.image]:

                # calculate how many pods of this service can be deployed while satisfying the max function factor
                reduce = max_replicas - self.functions_definitions[service.image]
                if reduce < 0:
                    # all replicas used
                    continue
                leftover_scale = leftover_scale - reduce
            if leftover_scale > 0:
                for _ in range(leftover_scale):
                    yield from self.deploy_replica(fd, fd.get_container(service.image), fd.get_containers()[index:])
                    actually_scaled += 1
                    scale -= 1

        self.env.metrics.log_scaling(fd.name, actually_scaled)

        if scale > 0:
            logger.debug("Function %s wanted to scale, but not all requested replicas were deployed: %s", fn_name,
                        str(scale))

    def next_replica(self, request) -> KFunctionReplica:
        return self.load_balancer.next_replica(request)

    def start(self):
        for process in self.env.background_processes:
            self.env.process(process(self.env))
        self.env.process(self.run_scheduler_worker())

    def poll_available_replica(self, fn: str, num_entry: int = 1,  interval:float = 0.5, state:AppState = AppState.LOADED_MODEL, max_retry:int = 60):
        retry = 0
        # Ta cần lấy các replica đang không bị lock
        while (len(self.get_replicas(fn, state)) < num_entry)  and (retry < max_retry):
            yield self.env.timeout(interval)
            retry += 1

    def run_scheduler_worker(self):
        env = self.env

        while True:
            replica: KFunctionReplica
            replica, services = yield self.scheduler_queue.get()

            logger.debug('scheduling next replica %s', replica.function.name)

            # schedule the required pod
            self.env.metrics.log_start_schedule(replica)
            pod = replica.pod
            # then = time.time()
            result = env.scheduler.schedule(pod)
            # duration = time.time() - then
            self.env.metrics.log_finish_schedule(replica, result)

            # yield env.timeout(duration)  # include scheduling latency in simulation time

            # if logger.isEnabledFor(logging.DEBUG):
            #     logger.debug('Pod scheduling took %.2f ms, and yielded %s', duration * 1000, result)

            if not result.suggested_host:
                self.replicas[replica.fn_name].remove(replica)
                if len(services) > 0:
                    logger.warning('retry scheduling pod %s', pod.name)
                    yield from self.deploy_replica(replica.function, services[0], services[1:])
                else:
                    logger.error('pod %s cannot be scheduled', pod.name)

                continue

            logger.info('pod %s was scheduled to %s', pod.name, result.suggested_host)

            replica.node = self.env.get_node_state(result.suggested_host.name)
            node = replica.node.skippy_node

            env.metrics.log('allocation', {
                'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
                'mem': 1 - (node.allocatable.memory / node.capacity.memory)
            }, node=node.name)

            self.functions_definitions[replica.image] += 1
            self.replica_count[replica.fn_name] += 1
            
            self.env.metrics.log_function_deploy(replica)
            # start a new process to simulate starting of pod
            env.process(simulate_function_start(env, replica))

    def create_pod(self, fd: FunctionDeployment, fn: FunctionContainer):
        return create_function_pod(fd, fn)

    def create_replica(self, fd: FunctionDeployment, fn: FunctionContainer) -> KFunctionReplica:
        replica = KFunctionReplica()
        replica.state = AppState.NULL
        replica.function = fd
        replica.container = fn
        replica.pod = self.create_pod(fd, fn)
        replica.locked = False
        replica.simulator = self.env.simulator_factory.create(self.env, fn)
        return replica

    # TODO: Vấn đề của cách làm hiện tại là sẽ có những container bị chuyển trạng thái dở dang khi api muốn nhảy nhiều bước
    # Cách fix hợp lý là logic chọn các relica để chuyển trạng thái nên do hàm change_state thực hiện.
    # Các hàm chuyển trạng thái nhận list này và thực hiện hành động
    def change_state(self, fn_name: str, num_replica: int, from_state: str, to_state: str):
        logger.info(f'Received request changing {num_replica} replicas of {fn_name} from {from_state} to {to_state} ')
        from_idx = self.states.index(from_state)
        to_idx = self.states.index(to_state)
        if from_idx < 0 or to_idx < 0:
            raise ValueError(f"Invalid state transition from {from_state} to {to_state}")
        if from_idx == to_idx:
            return
        
        if from_idx < to_idx:
            for i in range(from_idx, to_idx):
                yield from self.forward_transitions[i](fn_name, num_replica)
                
        else:
            for i in range(from_idx - 1, to_idx - 1, -1):
                yield from self.backward_transitions[i](fn_name, num_replica)

    def do_load_model(self, fn_name: str, num_replica: int = 0):
        # logger.debug(f'received request to load model for {num_replica} replicas of {fn_name}')
        running_replicas = self.get_replicas(fn_name, AppState.UNLOADED_MODEL)
        can_do = int(num_replica)
        
        if can_do == 0:
            return
        
        if len(running_replicas) < can_do:
            yield from self.poll_available_replica(fn=fn_name, num_entry=can_do,
                                                    interval=0.5, state=AppState.UNLOADED_MODEL, max_retry=60)
            
        running_replicas = self.get_replicas(fn_name, AppState.UNLOADED_MODEL)
        if len(running_replicas) < can_do:
            can_do = len(running_replicas)
            
        if can_do == 0:
            logger.info('Function %s wanted to load model, but no replicas were loaded', fn_name)
            return
        
        logger.info(f'loading model for {can_do} replica of {fn_name}')
        running_replicas = self.get_replicas(fn_name, AppState.UNLOADED_MODEL)
        replicas = running_replicas[len(running_replicas) - can_do:]
        
        self.env.metrics.log_load(fn_name, len(replicas))
        
        for replica in replicas:
            replica.locked = True
            
        for replica in replicas:
            yield from replica.simulator.load_model(self.env, replica)
            replica.locked = False
    
    def do_unload_model(self, fn_name: str, num_replica: int = 0):            
        running_replicas = self.get_replicas(fn_name, AppState.LOADED_MODEL)
        can_do = int(num_replica)
        if can_do == 0:
            return
        
        if len(running_replicas) < can_do:
            yield from self.poll_available_replica(fn=fn_name, num_entry=can_do,
                                                    interval=0.5, state=AppState.LOADED_MODEL, max_retry=60)
            
        running_replicas = self.get_replicas(fn_name, AppState.LOADED_MODEL)
        if len(running_replicas) < can_do:
            can_do = len(running_replicas)

        scale_min = self.functions_deployments[fn_name].scaling_config.scale_min
        if self.replica_count.get(fn_name, 0) - can_do < scale_min:
            can_do = self.replica_count.get(fn_name, 0) - scale_min

        if can_do == 0:
            logger.info('Function %s wanted to unload model, but no replicas were unloaded', fn_name)
            return

        logger.info(f'unload model for {can_do} replica of {fn_name}')
        running_replicas = self.get_replicas(fn_name, AppState.LOADED_MODEL)
        replicas = running_replicas[len(running_replicas) - can_do:]
        
        self.env.metrics.log_unload(fn_name, len(replicas))
        for replica in replicas:
            replica.locked = True
            
        for replica in replicas:
            yield from replica.simulator.unload_model(self.env, replica)
            replica.locked = False
            
def simulate_function_start(env: Environment, replica: KFunctionReplica):
    sim: FunctionSimulator = replica.simulator
    replica.locked = True
    logger.debug('deploying function %s to %s', replica.function.name, replica.node.name)
    env.metrics.log_deploy(replica)
    yield from sim.deploy(env, replica)
    env.metrics.log_startup(replica)
    logger.debug('starting function %s on %s', replica.function.name, replica.node.name)
    yield from sim.startup(env, replica)

    logger.debug('running function setup %s on %s', replica.function.name, replica.node.name)
    env.metrics.log_setup(replica)
    yield from sim.setup(env, replica)  # FIXME: this is really domain-specific startup
    env.metrics.log_finish_deploy(replica)
    replica.locked = False
    
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