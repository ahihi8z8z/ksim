import gymnasium as gym
import ksim_env 
import os
import numpy as np
import logging
import matplotlib.pyplot as plt

name_prefix = "scale_by_requests"


ksim_log_dir = f"logs/{name_prefix}/"
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
    pod_latency = []
    scaler_latency = []
    latency = []
    exec_interval = []

    for service, request in request_details.items():
        for req_id, details in request.items():
            l = 0
            if details.get('pod_latency', 0) > 0:
                l += details.get('pod_latency')
                pod_latency.append(details.get('pod_latency'))
            if details.get('scaler_latency', 0) > 0:
                l += details.get('scaler_latency')
                scaler_latency.append(details.get('scaler_latency', 0))
            if l > 0:
                latency.append(l)
            exec_interval.append(details.get('execution_time', 0))

    def compute_cdf(data):
        data = np.sort(data)
        cdf = np.arange(1, len(data) + 1) / len(data)
        return data, cdf

    x1, y1 = compute_cdf(latency)
    x2, y2 = compute_cdf(pod_latency)
    x3, y3 = compute_cdf(scaler_latency)
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=False)

    axs[0].plot(x1, y1, label='Latency', color='tab:blue')
    axs[0].set_title('CDF of Latency')
    axs[0].set_ylabel('CDF')
    axs[0].grid(True)

    axs[1].plot(x2, y2, label='Pod Latency', color='tab:blue')
    axs[1].set_title('CDF of Pod Latency')
    axs[1].set_ylabel('CDF')
    axs[1].grid(True)

    axs[2].plot(x3, y3, label='Scaler Latency', color='tab:blue')
    axs[2].set_title('CDF of Scaler Latency')
    axs[2].set_ylabel('CDF')
    axs[2].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, f"latency.png"))
    plt.close()
    return latency

def plot_logs(name_prefix, request_in, request_out, request_drop,
              ram_util, cpu_util,
              null, unloaded, loaded,
              reward_list, latency, power_util):

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

    axs[1].plot(smooth_list(power_util, window=60))
    axs[1].set_ylabel("Power Util (%)")
    axs[1].set_xlabel("Time Step")
    axs[1].grid(True)
    
    cpu_sum = rolling_sum(ram_util, window=288)
    req_sum = rolling_sum(request_out, window=288)

    # Tránh chia cho 0
    cpu_per_request = [c / r if r != 0 else 0 for c, r in zip(cpu_sum, req_sum)]

    axs[2].plot(cpu_per_request)
    axs[2].set_ylabel("RAM Util per request")
    axs[2].set_xlabel("Time Step")
    axs[2].grid(True)

    fig.suptitle("Smoothed RAM and Power Utilization")
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


env = gym.make('KsimEnv-V0', config_file="config_baseline.json", render_dir=f"logs/{name_prefix}")
obs = env.reset()
truncated = False
request_in = []
request_out = []
request_drop = []
ram_util = []
cpu_util = []
power_util = []
reward_list = []
null = []
unloaded = []
loaded = []

info = {}
while not truncated:
    action = np.zeros_like(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)
    request_in.append(info['request_in_over_step'])
    request_out.append(info['request_out_over_step'])
    request_drop.append(info['request_drop_over_step'])
    ram_util.append(info['ram_util'])
    cpu_util.append(info['cpu_util'])
    power_util.append(info['power_util'])
    reward_list.append(reward)
    null.append(info['NULL'])
    unloaded.append(info['UNLOADED_MODEL'])
    loaded.append(info['LOADED_MODEL'])

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
    latency=latency,
    power_util=power_util
)

