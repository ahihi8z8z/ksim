import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def smooth_list(list, window=60):
    if len(list) < window:
        return []
    return np.array([np.mean(list[i:i+window]) for i in range(len(list) - window + 1)])

def rolling_sum(data, window):
    return np.array([sum(data[i:i+window]) for i in range(len(data) - window + 1)])

def split_into_chunks(data, window):
    return np.array([np.array(data[i:i+window]) for i in range(0, len(data), window)])

def handle_request_details(request_details, log_dir: str = None):
    latency = []

    for service, details in request_details.items():
        latency = details.get('latency')

    latency_detail_path = f"{log_dir}/latency_detail.pkl"
    with open(latency_detail_path, 'wb') as f:
        pickle.dump(latency, f)
        
    plot_latency(latency)
    return latency_detail_path
 
def compute_cdf(data):
    data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)
    return data, cdf

def plot_latency(latency_detail_path, log_dir):
    with open(latency_detail_path, 'rb') as f:
        latency = pickle.load(f)
    x1, y1 = compute_cdf(latency)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x1, y1, label='Latency', color='tab:blue')
    ax.set_title('CDF of Latency')
    ax.set_ylabel('CDF')
    ax.set_xlabel('Latency (ms)')  # tuỳ vào đơn vị
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(log_dir, f"latency.png"))
    plt.close()

def plot_data(data_path, latency_detail_path, window_size, log_dir):
    data = pd.read_csv(data_path)
    
    request_in = data["request_in"].tolist()
    request_out = data["request_out"].tolist()
    request_drop = data["request_drop"].tolist()
    ram_util = data["ram_util"].tolist()
    power_util = data["power_util"].tolist()
    reward_list = data["reward_list"].tolist()
    null = data["null"].tolist()
    unloaded = data["unloaded"].tolist()
    loaded = data["loaded"].tolist()
    
    with open(latency_detail_path, "rb") as f:
        latency_detail = pickle.load(f)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(smooth_list(request_in, window=window_size))
    axs[0].set_ylabel("Request In")
    axs[0].grid(True)

    axs[1].plot(smooth_list(request_out, window=window_size))
    axs[1].set_ylabel("Request Out")
    axs[1].grid(True)

    axs[2].plot(smooth_list(request_drop, window=window_size))
    axs[2].set_ylabel("Request Drop")
    axs[2].set_xlabel("Time Step")
    axs[2].grid(True)

    fig.suptitle(f"Smoothed Requests In / Out / Drop ({window_size} pts)")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/requests.png")
    plt.close()


    labels = ['Dropped', 'Delayed', 'Non Delayed']
    sizes = [sum(request_drop), sum(latency_detail), sum(request_out) - sum(latency_detail)]
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
        plt.savefig(f"{log_dir}/request_components.png")
    plt.close()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(smooth_list(ram_util, window=window_size))
    axs[0].set_ylabel("RAM Util (%)")
    axs[0].grid(True)

    axs[1].plot(smooth_list(power_util, window=window_size))
    axs[1].set_ylabel("Power Util (%)")
    axs[1].set_xlabel("Time Step")
    axs[1].grid(True)
    
    ram_sum = rolling_sum(ram_util, window=window_size)
    req_sum = rolling_sum(request_out, window=window_size)
    ram_per_request = [c / r if r != 0 else 0 for c, r in zip(ram_sum, req_sum)]

    axs[2].plot(ram_per_request)
    axs[2].set_ylabel("RAM Util per request")
    axs[2].set_xlabel("Time Step")
    axs[2].grid(True)

    fig.suptitle(f"Smoothed RAM and CPU Utilization {window_size} pts")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/resource_util.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    # plt.plot(null, label='Null')
    plt.plot(smooth_list(unloaded, window=window_size), label='Unloaded')
    plt.plot(smooth_list(loaded, window=window_size), label='Loaded')
    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.title("Function States")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/function_states.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(smooth_list(reward_list, window=60), label=f"Smoothed Reward")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title(f"Smoothed Reward ({window_size} pts)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/reward.png")
    plt.close()
    
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