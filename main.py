import os
import json
import logging
import argparse
import ksim_env

import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from utils.plot_fig import handle_request_details, plot_data

# Cấu hình log 
log_config = json.load(open("log_config.json"))
log_prefix = log_config.get("log_prefix")
log_level = getattr(logging, log_config.get("log_level", "DEBUG").upper(), logging.INFO)
log_dir = f"logs/{log_prefix}/"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=log_level,
    format='[PID: %(process)d] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/ksim.log", mode='a')
    ],
    force=True
)

def train(args, system_config):
    global log_dir
    global log_prefix
    
    n_envs = args.n_envs
    n_steps = args.n_steps
    device = args.device
    total_timesteps = args.total_timesteps
    log_interval = args.log_interval
    eval_freq = args.eval_freq
    n_eval_episodes = args.n_eval_episodes
    deterministic = args.deterministic
    tensorboard_log = args.tensorboard_log
    
    train_env = make_vec_env(env_id="KsimEnv-V0", n_envs=n_envs, 
                        vec_env_cls=SubprocVecEnv, 
                        env_kwargs= {"system_config": system_config,
                                    "service_config": "service_config.json",}
                        )

    train_vec_env = VecMonitor(venv=train_env, 
                        filename=log_dir)  
    
    eval_callback = EvalCallback(train_vec_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=eval_freq, n_eval_episodes = n_eval_episodes,
                                deterministic=deterministic)

    model = PPO(policy="MultiInputPolicy", n_steps=n_steps, 
                env=train_vec_env, 
                verbose=2, device=device,
                tensorboard_log=tensorboard_log)
    
    model.learn(total_timesteps=total_timesteps, 
                callback=eval_callback, 
                log_interval=log_interval, progress_bar=True, 
                tb_log_name=log_prefix) 
    
    model.save(f"{log_dir}/ksim_model")

def evaluate(args, system_config):
    n_envs = args.n_envs
    model_path = args.model_path
    device = args.device
    window_size = args.window_size
    deterministic = args.deterministic
    
    eval_env = make_vec_env(env_id="KsimEnv-V0", n_envs=n_envs, 
                        vec_env_cls=SubprocVecEnv, 
                        env_kwargs= {"system_config": system_config,
                                    "service_config": "service_config.json",}
                        )
    model = PPO.load(path=model_path, env=eval_env, device=device)
    env = model.get_env()
    obs = env.reset()
    template = {
        "request_in": [],
        "request_out": [],
        "request_drop": [],
        "ram_util": [],
        "power_util": [],
        "reward_list": [],
        "null": [],
        "unloaded": [],
        "loaded": []
    }
    
    data = [ {key: [] for key in template} for _ in range(n_envs) ]
    dones = [False]*n_envs
    infos = {}
    
    while not any(dones):
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)
        data_collection(data, rewards, infos)
        
    for i in range(n_envs):
        env_log_dir = os.path.join(log_dir, f"env_{i}")
        os.makedirs(env_log_dir, exist_ok=True)
        env_data_path = f"{env_log_dir}/data.csv"
        df = pd.DataFrame(data[i])
        df.to_csv(env_data_path, index=False)
    
        latency_detail_path = handle_request_details(infos[i].get("request_details", {}), env_log_dir)
        plot_data(env_data_path, latency_detail_path, window_size=window_size, log_dir=env_log_dir)

def baseline(args, system_config):
    n_envs = 1
    window_size = args.window_size
    
    env = gym.make('KsimEnv-V0', system_config=system_config, service_config="service_config.json")
    obs = env.reset()
    truncated = False
    template = {
        "request_in": [],
        "request_out": [],
        "request_drop": [],
        "ram_util": [],
        "power_util": [],
        "reward_list": [],
        "null": [],
        "unloaded": [],
        "loaded": []
    }
    data = [ {key: [] for key in template} for _ in range(n_envs) ]

    info = {}
    while not truncated:
        action = np.zeros_like(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        data_collection(data, [reward], [info])

    for i in range(n_envs):
        env_log_dir = os.path.join(log_dir, f"env_{i}")
        os.makedirs(env_log_dir, exist_ok=True)
        env_data_path = f"{env_log_dir}/data.csv"
        df = pd.DataFrame(data[i])
        df.to_csv(env_data_path, index=False)
    
        latency_detail_path = handle_request_details(info.get("request_details", {}), env_log_dir)
        plot_data(env_data_path, latency_detail_path, window_size=window_size, log_dir=env_log_dir)

def data_collection(data, rewards, info):
    i=0
    for env_data in data:
        env_data["request_in"].append(info[i]['request_in_over_step'])
        env_data["request_out"].append(info[i]['request_out_over_step'])
        env_data["request_drop"].append(info[i]['request_drop_over_step'])
        env_data["ram_util"].append(info[i]['ram_util'])
        env_data["power_util"].append(info[i]['power_util'])
        env_data["reward_list"].append(rewards[i])
        env_data["null"].append(info[i]['NULL'])
        env_data["unloaded"].append(info[i]['UNLOADED_MODEL'])
        env_data["loaded"].append(info[i]['LOADED_MODEL'])
        i+=1

def main(args):
    global log_dir
    mode = args.mode
    
    with open("system_config.json", 'r') as f:
        system_config_all = json.load(f)
        
    system_config = system_config_all["general"]
    system_config.update(system_config_all[mode])
    
    if args.clear_logs:
        log_file = f"{log_dir}/ksim.log"
        if os.path.exists(log_file):
            with open(log_file, 'w'):
                pass
            
    if mode == "train":
        train(args, system_config)
    elif mode == "eval":
        evaluate(args, system_config)
    elif mode == "baseline":
        baseline(args, system_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running faas")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "baseline"], 
        required=True,
        help="Choose one of: train, eval, baseline"
    )

    parser.add_argument("--n-envs", type=int, default=8, help="Number of environments to run in parallel")
    parser.add_argument("--eval-freq", type=int, default=288, help="Frequency of evaluation in timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=1, help="Number of episodes for evaluation")
    parser.add_argument("--deterministic", type=bool, default=True, help="Run evaluation in deterministic mode")
    parser.add_argument("--render", type=bool, default=False, help="Render the environment during evaluation")
    parser.add_argument("--tensorboard-log", type=str, default="./ksim_tensorboard/", help="TensorBoard log directory")
    parser.add_argument("--n-steps", type=int, default=32, help="Number of steps per update")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--total-timesteps", type=int, default=228*14*30, help="Total timesteps for training")
    parser.add_argument("--log-interval", type=int, default=1, help="Logging interval")
    parser.add_argument("--model-path", type=str, default="logs/PPO/best_model.zip", help="Path to the model for evaluation")
    parser.add_argument("--window-size", type=int, default=1, help="Window size for smoothing in plots")
    parser.add_argument("--clear-logs", type=bool, default=True, help="Clear existing logs before running")

    args = parser.parse_args()
    main(args)