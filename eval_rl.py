import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import ksim_env 
import os
import numpy as np

import logging
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env, unwrap_wrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

name_prefix = "eval_ppo"

ksim_log_dir = f"./logs/{name_prefix}/"
os.makedirs(ksim_log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[PID: %(process)d] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{ksim_log_dir}/ksim.log", mode='a') 
    ]
)

logger = logging.getLogger(__name__)

def smooth_list(list, window=60):
    if len(list) < window:
        return []
    return np.array([np.mean(list[i:i+window]) for i in range(len(list) - window + 1)])

def rolling_sum(data, window):
    return np.array([sum(data[i:i+window]) for i in range(len(data) - window + 1)])

def split_into_chunks(data, window):
    return np.array([np.array(data[i:i+window]) for i in range(0, len(data), window)])

def plot_request_details(request_details, log_dir: str = None):
    latency = []
    exec_interval = []

    for service, request in request_details.items():
        for req_id, details in request.items():
            if details.get('pod_latency', 0) + details.get('scaler_latency', 0) > 0:
                latency.append(details.get('pod_latency', 0) + details.get('scaler_latency', 0))
            exec_interval.append(details.get('execution_time', 0))

    def compute_cdf(data):
        data = np.sort(data)
        cdf = np.arange(1, len(data) + 1) / len(data)
        return data, cdf

    x1, y1 = compute_cdf(latency)

    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=False)

    axs[0].plot(x1, y1, label='Latency', color='tab:blue')
    axs[0].set_title('CDF of Latency')
    axs[0].set_ylabel('CDF')
    axs[0].grid(True)

    axs[1].hist(latency, bins=100, density=True, color='tab:green', edgecolor='black')
    axs[1].set_title('PDF of Latency (100 bins)')
    axs[1].set_xlabel('Latency (seconds)')
    axs[1].set_ylabel('Density')
    axs[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, f"latency.png"))
    plt.close()
    return latency

def plot_logs(name_prefix, request_in, request_out, request_drop,
              ram_util, cpu_util,
              null, unloaded, loaded,
              reward_list, latency):

    os.makedirs(f"logs/{name_prefix}", exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(smooth_list(request_in, window=60))
    axs[0].set_ylabel("Request In")
    axs[0].grid(True)

    axs[1].plot(smooth_list(request_out, window=60))
    axs[1].set_ylabel("Request Out")
    axs[1].grid(True)

    axs[2].plot(smooth_list(request_drop, window=60))
    axs[2].set_ylabel("Request Drop")
    axs[2].set_xlabel("Time Step")
    axs[2].grid(True)

    fig.suptitle("Smoothed Requests In / Out / Drop (60 pts)")
    plt.tight_layout()
    plt.savefig(f"logs/{name_prefix}/requests.png")
    plt.close()


    labels = ['Dropped', 'Delayed', 'Non Delayed']
    sizes = [sum(request_drop), sum(latency), sum(request_out) - sum(latency)]
    filtered = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    labels, sizes = zip(*filtered) if filtered else ([], [])
    def make_autopct(values):
        def autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{val}\n({pct:.1f}%)"
        return autopct
    plt.figure(figsize=(6, 6))
    if sizes:
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            autopct=make_autopct(sizes),
            startangle=90,
            labeldistance=0.7,
            textprops=dict(color="white", fontsize=10)
        )
        plt.axis('equal')
        plt.title('Request Components')
        plt.tight_layout()
        plt.savefig(f"logs/{name_prefix}/request_components.png")
    plt.close()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(smooth_list(ram_util, window=60))
    axs[0].set_ylabel("RAM Util (%)")
    axs[0].grid(True)

    axs[1].plot(smooth_list(cpu_util, window=60))
    axs[1].set_ylabel("CPU Util (%)")
    axs[1].set_xlabel("Time Step")
    axs[1].grid(True)
    
    cpu_sum = rolling_sum(cpu_util, window=288)
    req_sum = rolling_sum(request_out, window=288)

    # Tránh chia cho 0
    cpu_per_request = [c / r if r != 0 else 0 for c, r in zip(cpu_sum, req_sum)]

    axs[2].plot(cpu_per_request)
    axs[2].set_ylabel("CPU Util per request")
    axs[2].set_xlabel("Time Step")
    axs[2].grid(True)

    fig.suptitle("Smoothed RAM and CPU Utilization")
    plt.tight_layout()
    plt.savefig(f"logs/{name_prefix}/resource_util.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    # plt.plot(null, label='Null')
    plt.plot(smooth_list(unloaded, window=60), label='Unloaded')
    plt.plot(smooth_list(loaded, window=60), label='Loaded')
    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.title("Function States")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"logs/{name_prefix}/function_states.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(smooth_list(reward_list, window=60), label='Smoothed Reward (60 pts)')
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Smoothed Reward (60 pts)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"logs/{name_prefix}/reward.png")
    plt.close()


if __name__=="__main__":
    eval_env = make_vec_env(env_id="KsimEnv-V0", n_envs=1, 
                        vec_env_cls=SubprocVecEnv, 
                        env_kwargs= {"config_file": "config_rl_eval.json",
                                    "render_dir": f"logs/{name_prefix}"})
    model = PPO.load(path="/home/haipt/ksim/logs/PPO/best_model.zip",
                     env=eval_env, device="cpu")
    env = model.get_env()
    obs = env.reset()
    truncated = [False]
    request_in = []
    request_out = []
    request_drop = []
    ram_util = []
    cpu_util = []
    reward_list = []
    null = []
    unloaded = []
    loaded = []

    info = {}
    while not truncated[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        request_in.append(info[0]['request_in_over_step'])
        request_out.append(info[0]['request_out_over_step'])
        request_drop.append(info[0]['request_drop_over_step'])
        ram_util.append(info[0]['ram_util'])
        cpu_util.append(info[0]['cpu_util'])
        reward_list.append(rewards[0])
        null.append(info[0]['NULL'])
        unloaded.append(info[0]['UNLOADED_MODEL'])
        loaded.append(info[0]['LOADED_MODEL'])
        
    latency = plot_request_details(info["request_details"], f"logs/{name_prefix}")

    plot_logs(
        name_prefix=name_prefix, 
        request_in=request_in,
        request_out=request_out,
        request_drop=request_drop,
        ram_util=ram_util,
        cpu_util=cpu_util,
        null=null,
        unloaded=unloaded,
        loaded=loaded,
        reward_list=reward_list,
        latency=latency
    )