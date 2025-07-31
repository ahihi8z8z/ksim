from sim.core import Environment
from sim.faas import FunctionContainer, SimulatorFactory, FunctionRequest, FunctionReplica, FunctionSimulator
from sim import docker
from ksim_env.utils.metrics import KFunctionResourceUsage
from ksim_env.utils.appstate import AppState

import simpy

class KSimulatorFactory(SimulatorFactory):
    def __init__(self, service_profile: dict) -> None:
        super().__init__()
        self.service_profile = service_profile

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        service_profile = self.service_profile.get(fn.image)
        # 1 replica chỉ xử lý 1 request tại 1 thời điểm
        return KFunctionSimulator(service_profile=service_profile)

class KFunctionSimulator(FunctionSimulator):
    queue: simpy.Resource
    def __init__(self, service_profile: dict) -> None:
        self.workers = service_profile.get("num_workers")
        self.queue = None
        self.service_profile = service_profile.get("state_resource_usage")
        self.changing_state = False
        
    def claim_resources(self, env: Environment, replica: FunctionReplica, usage: KFunctionResourceUsage):
        env.resource_state.put_resource(replica, 'cpu', usage.cpu)
        env.resource_state.put_resource(replica, 'memory', usage.ram)
        yield env.timeout(usage.time) 

    def release_resources(self, env: Environment, replica: FunctionReplica, usage: KFunctionResourceUsage):
        env.resource_state.remove_resource(replica, 'cpu', usage.cpu)
        env.resource_state.remove_resource(replica, 'memory', usage.ram)
    
    def deploy(self, env: Environment, replica: FunctionReplica):
        # simulate a docker pull command for deploying the function (also done by sim.faassim.DockerDeploySimMixin)
        replica.state = AppState.STARTING
        replica.locked = True
        yield from docker.pull(env, replica.container.image, replica.node.ether_node)
        replica.locked = False
        

    def startup(self, env: Environment, replica: FunctionReplica):
        self.queue = simpy.Resource(env, capacity=self.workers)
        
        replica.state = AppState.STARTING
        replica.locked = True
        yield from self.claim_resources(env, replica, self.service_profile[AppState.STARTING])
        self.release_resources(env, replica, self.service_profile[AppState.STARTING])
        
        yield from self.claim_resources(env, replica, self.service_profile[AppState.UNLOADED_MODEL])
        replica.state = AppState.UNLOADED_MODEL
        replica.locked = False
        

    def teardown(self, env: Environment, replica: FunctionReplica):
        self.release_resources(env, replica, self.service_profile[AppState.UNLOADED_MODEL])

        replica.state = AppState.SUSPENDED
        replica.locked = True
        yield from self.claim_resources(env, replica, self.service_profile[AppState.SUSPENDED])
        self.release_resources(env, replica, self.service_profile[AppState.SUSPENDED])
        
        yield from self.claim_resources(env, replica, self.service_profile[AppState.NULL])
        replica.state = AppState.NULL
        replica.locked = True
        
    def load_model(self, env: Environment, replica: FunctionReplica):
        self.release_resources(env, replica, self.service_profile[AppState.UNLOADED_MODEL])

        replica.state = AppState.LOADING_MODEL
        replica.locked = True
        yield from self.claim_resources(env, replica, self.service_profile[AppState.LOADING_MODEL])
        self.release_resources(env, replica, self.service_profile[AppState.LOADING_MODEL])
        
        yield from self.claim_resources(env, replica, self.service_profile[AppState.LOADED_MODEL])
        replica.state = AppState.LOADED_MODEL
        replica.locked = False
        
    def unload_model(self, env: Environment, replica: FunctionReplica):
        self.release_resources(env, replica, self.service_profile[AppState.LOADED_MODEL])

        replica.state = AppState.UNLOADING_MODEL
        replica.locked = True
        yield from self.claim_resources(env, replica, self.service_profile[AppState.UNLOADING_MODEL])
        self.release_resources(env, replica, self.service_profile[AppState.UNLOADING_MODEL])
        
        yield from self.claim_resources(env, replica, self.service_profile[AppState.UNLOADED_MODEL])
        replica.state = AppState.UNLOADED_MODEL
        replica.locked = False

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        token = self.queue.request()
        # t_wait_start = env.now
        yield token
        # t_wait_end = env.now

        # t_fet_start = env.now
        # logging.debug('invoking function %s on node %s', request, replica.node.name)

        replica.node.current_requests.add(request)

        replica.state = AppState.ACTIVING
        replica.locked = True
        env.resource_state.put_resource(replica, 'cpu', self.service_profile[AppState.ACTIVING].cpu)
        env.resource_state.put_resource(replica, 'memory', self.service_profile[AppState.ACTIVING].ram)
        yield env.timeout(request.size)
        env.resource_state.remove_resource(replica, 'cpu', self.service_profile[AppState.ACTIVING].cpu)
        env.resource_state.remove_resource(replica, 'memory', self.service_profile[AppState.ACTIVING].ram)
        
        replica.state = AppState.LOADED_MODEL
        replica.locked = False

        # t_fet_end = env.now

        replica.node.current_requests.remove(request)

        # env.metrics.log_fet(replica.fn_name, replica.image, replica.node.name,
        #                     t_fet_start=t_fet_start, t_fet_end=t_fet_end, replica_id=id(replica),
        #                     request_id=request.request_id,
        #                     t_wait_start=t_wait_start, t_wait_end=t_wait_end)

        self.queue.release(token)