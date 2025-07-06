import logging
import math

import numpy as np

from sim.core import Environment
from sim.faas import FaasSystem, FunctionDeployment
from ksim_env.utils.appstate import AppState
from ksim_env.utils.monitor import KMetricsServer

logger = logging.getLogger(__name__)

class FaasRequestScaler:

    def __init__(self, fn: FunctionDeployment, env: Environment):
        self.env = env
        self.function_invocations = dict()
        self.reconcile_interval = fn.scaling_config.rps_threshold_duration
        self.threshold = fn.scaling_config.rps_threshold
        self.alert_window = fn.scaling_config.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: FaasSystem = env.faas
        while self.running:
            yield env.timeout(self.reconcile_interval)
            if self.function_invocations.get(self.fn_name, None) is None:
                self.function_invocations[self.fn_name] = 0
            last_invocations = self.function_invocations.get(self.fn_name, 0)
            current_total_invocations = env.metrics.invocations.get(self.fn_name, 0)
            invocations = current_total_invocations - last_invocations
            self.function_invocations[self.fn_name] += invocations
            # TODO divide by alert window, but needs to store the invocations, such that reconcile_interval != alert_window is possible
            config = self.fn.scaling_config
            logger.debug(f'invocations {invocations} and reconcile_interval {self.reconcile_interval} and threshold {self.threshold}')
            if (invocations / self.reconcile_interval) >= self.threshold:
                scale = (config.scale_factor / 100) * config.scale_max
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "NULL", "LOADED_MODEL"))
                logger.debug(f'scaler scaled up and loaded model {self.fn_name} by {int(np.ceil(scale))}')       
            else:
                scale = (config.scale_factor / 100) * config.scale_max
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "LOADED_MODEL",  "NULL"))
                logger.debug(f'scaler unloaded model and scaled down {self.fn_name} by {int(np.ceil(scale))}')

    def stop(self):
        self.running = False


class AverageFaasRequestScaler:
    """
    Scales deployment according to the average number of requests distributed equally over all replicas.
    The distributed property holds as per default the round robin scheduler is used
    """

    def __init__(self, fn: FunctionDeployment, env: Environment):
        self.env = env
        self.function_invocations = dict()
        self.threshold = fn.scaling_config.target_average_rps
        self.alert_window = fn.scaling_config.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: FaasSystem = env.faas
        while self.running:
            yield env.timeout(self.alert_window)
            if self.function_invocations.get(self.fn_name, None) is None:
                self.function_invocations[self.fn_name] = 0
                
            running = len(faas.get_replicas(self.fn.name, AppState.LOADED_MODEL, False))
            running += len(faas.get_replicas(self.fn.name, AppState.ACTIVING, False))
            
            if running == 0:
                logger.debug("scaler 0 running replica")
                continue

            starting_replicas = len(faas.get_replicas(self.fn.name, AppState.STARTING, False))
            starting_replicas += len(faas.get_replicas(self.fn.name, AppState.UNLOADED_MODEL, False))
            starting_replicas += len(faas.get_replicas(self.fn.name, AppState.LOADING_MODEL, False))

            last_invocations = self.function_invocations.get(self.fn_name, 0)
            current_total_invocations = env.metrics.invocations.get(self.fn_name, 0)
            invocations = current_total_invocations - last_invocations
            self.function_invocations[self.fn_name] += invocations
            average = invocations / running
            desired_replicas = math.ceil(running * (average / self.threshold))

            updated_desired_replicas = desired_replicas
            if starting_replicas > 0:
                if desired_replicas > running:
                    count = running + starting_replicas
                    average = invocations / count
                    updated_desired_replicas = math.ceil(running * (average / self.threshold))

            if desired_replicas > running and updated_desired_replicas < running:
                logger.debug(f"scaler no scaling in case of reversed decision: desired_replicas-{desired_replicas}, updated_desired_replicas-{updated_desired_replicas}, running-{running}")
                continue

            ratio = average / self.threshold
            if 1 > ratio >= 1 - self.fn.scaling_config.target_average_rps_threshold:
                logger.debug(f"scaler ratio {ratio} is sufficiently close to 1.0")
                continue

            if 1 < ratio < 1 + self.fn.scaling_config.target_average_rps_threshold:
                logger.debug(f"scaler ratio {ratio} is sufficiently close to 1.0")
                continue

            logger.debug(f'scaler desired_replicas {desired_replicas} and running {running}')
            if desired_replicas < running:
                # scale down
                scale = running - desired_replicas
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "LOADED_MODEL",  "NULL"))
                logger.debug(f'scaler unloaded model and scaled down {self.fn_name} by {int(np.ceil(scale))}')
            else:
                # scale up
                scale = desired_replicas - running
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "NULL", "LOADED_MODEL"))
                logger.debug(f'scaler scaled up and loaded model {self.fn_name} by {int(np.ceil(scale))}') 

    def stop(self):
        self.running = False


class AverageQueueFaasRequestScaler:
    """
    Scales deployment according to the average number of requests distributed equally over all replicas.
    The distributed property holds as per default the round robin scheduler is used
    """

    def __init__(self, fn: FunctionDeployment, env: Environment):
        self.env = env
        self.threshold = fn.scaling_config.target_queue_length
        self.alert_window = fn.scaling_config.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: FaasSystem = env.faas
        while self.running:
            yield env.timeout(self.alert_window)
            running_replicas = faas.get_replicas(self.fn.name, AppState.LOADED_MODEL, False)
            running_replicas += faas.get_replicas(self.fn.name, AppState.ACTIVING, False)
            running = len(running_replicas)
            
            if running == 0:
                continue

            starting_replicas = len(faas.get_replicas(self.fn.name, AppState.STARTING, False))
            starting_replicas += len(faas.get_replicas(self.fn.name, AppState.UNLOADED_MODEL, False))
            starting_replicas += len(faas.get_replicas(self.fn.name, AppState.LOADING_MODEL, False))

            in_queue = []
            for replica in running_replicas:
                sim = replica.simulator
                in_queue.append(len(sim.queue.queue))
                
            if len(in_queue) == 0:
                average = 0
            else:
                average = int(math.ceil(np.median(np.array(in_queue))))

            desired_replicas = math.ceil(running * (average / self.threshold))

            updated_desired_replicas = desired_replicas
            if starting_replicas:
                if desired_replicas > running:
                    for _ in range(starting_replicas):
                        in_queue.append(0)

                    average = int(math.ceil(np.median(np.array(in_queue))))
                    updated_desired_replicas = math.ceil(running * (average / self.threshold))

            if desired_replicas > running and updated_desired_replicas < running:
                # no scaling in case of reversed decision
                continue

            ratio = average / self.threshold
            if 1 > ratio >= 1 - self.fn.scaling_config.target_average_rps_threshold:
                # ratio is sufficiently close to 1.0
                continue

            if 1 < ratio < 1 + self.fn.scaling_config.target_average_rps_threshold:
                continue

            if desired_replicas < running:
                # scale down
                scale = running - desired_replicas
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "LOADED_MODEL",  "NULL"))
                logger.debug(f'scaler unloaded model and scaled down {self.fn_name} by {int(np.ceil(scale))}')
            else:
                # scale up
                scale = desired_replicas - running
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "NULL", "LOADED_MODEL"))
                logger.debug(f'scaler scaled up and loaded model {self.fn_name} by {int(np.ceil(scale))}') 

    def stop(self):
        self.running = False
        
class HorizontalPodAutoscaler:

    # Behavior and default values taken from:
    # https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
    # Official implementation only considers pods, in which all containers have specified resource requests

    def __init__(self, env: Environment, average_window: int = 100, reconcile_interval: int = 15,
                 target_tolerance: float = 0.1):
        """
        :param average_window: seconds to look back in time to calculate the average for each replica
        :param env: sim environment
        :param reconcile_interval: wait time for control loop
        :param target_tolerance: determines how close the target/current resource ratio must be to 1.0 to skip scaling
        """
        self.average_window = 100
        self.env = env
        self.reconcile_interval = reconcile_interval
        self.target_tolerance = target_tolerance

    def run(self):
        """
        For each Function Deployment sum up the CPU usage of each running replica and take the mean.

        While the official implementation just uses the CPU usage reported by the metrics server,
        there is no option/default for the corresponding utilization window.
        Therefore, this implementation allows users to set the window size that will be used when querying the
        MetricsServer

        This implementation considers the target value to be a 'targetAverageUtilization', because our MetricsServer
        only can calculate the average utilization of on replica and has no means to report exact values (i.e.: millis)
        Also, the official HPA calculates average relative to the requested resource.
        We do this relative to the maximum (1.0 == 100% Utilization of all cores)

        Further, copied from docs:
        "If there were any missing metrics, we recompute the average more conservatively, assuming those pods
        were consuming 100% of the desired value in case of a scale down, and 0% in case of a scale up.
        This dampens the magnitude of any potential scale <-(!) not implemented, as "missing metrics" can't exist in sim

        Furthermore, if any not-yet-ready pods were present, and we would have scaled up without factoring
        in missing metrics  or not-yet-ready pods, we conservatively assume the not-yet-ready pods are consuming 0%
        of the desired metric, further dampening the magnitude of a scale up. <- implemented, considering: conceived
        and starting nodes to be 'not-yet-ready'.

        After factoring in the not-yet-ready pods and missing metrics, we recalculate the usage ratio. If the new ratio
        reverses the scale direction, or is within the tolerance, we skip scaling. Otherwise, we use the new ratio
        to scale. <- implemented"

        Raw Calculation:
        desiredReplicas = ceil[currentReplicas * ( currentMetricValue / desiredMetricValue )]
        """
        while True:
            yield self.env.timeout(self.reconcile_interval)
            metrics_server: KMetricsServer = self.env.metrics_server
            faas: FaasSystem = self.env.faas
            for function_deployment in faas.get_deployments():
                running_replicas = faas.get_replicas(self.fn.name, AppState.LOADED_MODEL, False)
                running_replicas += faas.get_replicas(self.fn.name, AppState.ACTIVING, False)
                running = len(running_replicas)
                if running == 0:
                    continue
                
                starting_replicas = len(faas.get_replicas(self.fn.name, AppState.STARTING, False))
                starting_replicas += len(faas.get_replicas(self.fn.name, AppState.UNLOADED_MODEL, False))
                starting_replicas += len(faas.get_replicas(self.fn.name, AppState.LOADING_MODEL, False))
                sum_cpu = 0

                for replica in running_replicas:
                    sum_cpu += metrics_server.get_average_cpu_utilization(replica, self.average_window)

                average_cpu = sum_cpu / running

                target_avg_utilization = function_deployment.scaling_config.target_average_utilization
                desired_replicas = math.ceil(
                    running * (average_cpu / target_avg_utilization))

                updated_desired_replicas = desired_replicas
                if starting_replicas:
                    if desired_replicas > running:
                        count = running + starting_replicas
                        average_cpu = sum_cpu / count
                        updated_desired_replicas = math.ceil(
                            running * (average_cpu / target_avg_utilization))

                if desired_replicas > running and updated_desired_replicas < running:
                    # no scaling in case of reversed decision
                    continue

                ratio = average_cpu / target_avg_utilization
                if 1 > ratio >= 1 - self.target_tolerance:
                    # ratio is sufficiently close to 1.0
                    continue

                if 1 < ratio < 1 + self.target_tolerance:
                    continue

                if desired_replicas < running:
                    # scale down
                    scale = running - desired_replicas
                    self.env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "LOADED_MODEL",  "NULL"))
                    logger.debug(f'scaler unloaded model and scaled down {self.fn_name} by {int(np.ceil(scale))}')
                else:
                    # scale up
                    scale = desired_replicas - running
                    self.env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "NULL", "LOADED_MODEL"))
                    logger.debug(f'scaler scaled up and loaded model {self.fn_name} by {int(np.ceil(scale))}') 
