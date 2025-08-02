import logging
import math

import numpy as np
from typing import List

from sim.core import Environment
from sim.faas import FaasSystem, FunctionDeployment
from ksim_env.utils.appstate import AppState
from ksim_env.utils.monitor import KMetricsServer

import torch
import joblib


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
            logging.debug(f'invocations {invocations} and reconcile_interval {self.reconcile_interval} and threshold {self.threshold}')
            if (invocations / self.reconcile_interval) >= self.threshold:
                scale = (config.scale_factor / 100) * config.scale_max
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "NULL", "LOADED_MODEL"))
                logging.debug(f'scaler scaled up and loaded model {self.fn_name} by {int(np.ceil(scale))}')       
            else:
                scale = (config.scale_factor / 100) * config.scale_max
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "LOADED_MODEL",  "NULL"))
                logging.debug(f'scaler unloaded model and scaled down {self.fn_name} by {int(np.ceil(scale))}')

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
                logging.debug("scaler 0 running replica")
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
                logging.debug(f"scaler no scaling in case of reversed decision: desired_replicas-{desired_replicas}, updated_desired_replicas-{updated_desired_replicas}, running-{running}")
                continue

            ratio = average / self.threshold
            if 1 > ratio >= 1 - self.fn.scaling_config.target_average_rps_threshold:
                logging.debug(f"scaler ratio {ratio} is sufficiently close to 1.0")
                continue

            if 1 < ratio < 1 + self.fn.scaling_config.target_average_rps_threshold:
                logging.debug(f"scaler ratio {ratio} is sufficiently close to 1.0")
                continue

            logging.debug(f'scaler desired_replicas {desired_replicas} and running {running}')
            if desired_replicas < running:
                # scale down
                scale = running - desired_replicas
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "LOADED_MODEL",  "NULL"))
                logging.debug(f'scaler unloaded model and scaled down {self.fn_name} by {int(np.ceil(scale))}')
            else:
                # scale up
                scale = desired_replicas - running
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "NULL", "LOADED_MODEL"))
                logging.debug(f'scaler scaled up and loaded model {self.fn_name} by {int(np.ceil(scale))}') 

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
                logging.debug(f'scaler unloaded model and scaled down {self.fn_name} by {int(np.ceil(scale))}')
            else:
                # scale up
                scale = desired_replicas - running
                env.process(faas.change_state(self.fn_name, int(np.ceil(scale)), "NULL", "LOADED_MODEL"))
                logging.debug(f'scaler scaled up and loaded model {self.fn_name} by {int(np.ceil(scale))}') 

    def stop(self):
        self.running = False

class LSTMScaler:
    """
    https://faculty.washington.edu/wlloyd/courses/tcss591/papers/Mitigating_Cold_Start_Problem_in_Serverless_Computing_A_Reinforcement_Learning_Approach.pdf
    Bài này dùng cho OpenWhisk, workaround thành scaler cho ksim.
    Cái làm 2 việc: 
    1. Scale down khi idle quá thời gian idle_threshold
    2. Scale up dựa vào dự đoán của LSTM model về số lượng request trong alert_window tiếp theo.
    """

    def __init__(self, fn: FunctionDeployment, env: Environment):
        self.env = env
        self.function_invocations = dict()
        self.idle_threshold = 60
        self.lstm_factor = fn.scaling_config.lstm_factor
        self.lstm_sequence_length = fn.scaling_config.lstm_sequence_length

        # Load PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = self._load_model(fn.scaling_config.lstm_model_path).to(self.device)
        self.lstm_model.eval()

        self.scaler = joblib.load(fn.scaling_config.scaler_path)

        self.alert_window = fn.scaling_config.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def _load_model(self, model_path):
        class LSTMModel(torch.nn.Module):
            def __init__(self, input_size=1, hidden_size=32, num_layers=4, dropout=0.5):
                super(LSTMModel, self).__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                    dropout=dropout, batch_first=True)
                self.fc = torch.nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                out = self.fc(out)
                return out

        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def run(self):
        env: Environment = self.env  
        faas: FaasSystem = env.faas
        
        while self.running:
            yield env.timeout(self.alert_window)
            now = env.now

            needed_replicas = self.predict_needed_replicas()
            running = faas.get_replicas(self.fn.name, AppState.LOADED_MODEL, False)
            if needed_replicas > len(running):
                scale = needed_replicas - len(running)
                env.process(faas.change_state(self.fn_name, scale, "NULL", "LOADED_MODEL"))
                logging.info(f'LSTMScaler want to scale up {self.fn_name} by {scale}')
            else:
                redundant = len(running) - needed_replicas
                scaled_down = 0
                specific_replicas = []
                for replica in running:
                    if replica.last_invocation < now - self.idle_threshold and scaled_down < redundant:
                        specific_replicas.append(replica)
                        scaled_down += 1
                env.process(faas.change_state(self.fn_name, 1, "LOADED_MODEL", "NULL", specific_replicas=specific_replicas))
                logging.info(f'LSTMScaler want to scale down {self.fn_name} by {redundant} replicas and {scaled_down} scaled')

    def predict_needed_replicas(self):
            labels = {"function": self.fn.name}
            now = self.env.now
            values = self.env.metrics_server.get_series("request_in", labels, now - self.alert_window*self.lstm_sequence_length, self.alert_window, now)
            values = self._fill_and_pad(values)
            input = np.array(values).reshape(-1, 1)
            scaled_input = self.scaler.transform(input).reshape(1, -1, 1)
            input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                pred_tensor = self.lstm_model(input_tensor)
            pred_rescaled = self.scaler.inverse_transform(pred_tensor.cpu().numpy())
            needed_replicas = int(np.ceil(pred_rescaled[0, 0] * self.lstm_factor / self.alert_window ))
            needed_replicas = max(0, needed_replicas)
            return needed_replicas
        
    def _fill_and_pad(self, values: List[float]) -> List[float]:
        if len(values) == 0:
            return [0]*self.lstm_sequence_length

        if len(values) < self.lstm_sequence_length:
            pad_val = values[-1]
            pad_len = self.lstm_sequence_length - len(values)
            values = values + [pad_val] * pad_len
        else:
            values = values[-self.lstm_sequence_length:]

        return values
    
    def stop(self):
        self.running = False
        
    def config_threshold(self, idle_threshold: float):
        self.idle_threshold = idle_threshold
        logging.info(f"LSTMScaler threshold updated to {self.idle_threshold}")