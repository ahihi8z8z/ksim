import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import ksim_env 

import logging
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env, unwrap_wrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

logging.basicConfig(
    level=logging.INFO,
    format='[PID: %(process)d] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ksim.log", mode='a') 
    ]
)

name_prefix = "ppo2"
logger = logging.getLogger(__name__)

if __name__=="__main__":
    train_env = make_vec_env(env_id="KsimEnv-V0", n_envs=12, 
                        vec_env_cls=SubprocVecEnv, 
                        # wrapper_class=TimeLimit,
                        # wrapper_kwargs={"max_episode_steps": 144}
                        env_kwargs= {"config_file": "config.json"})
    
    # eval_env = make_vec_env(env_id="KsimEnv-V0", n_envs=1, 
    #                     # vec_env_cls=SubprocVecEnv, 
    #                     wrapper_class=TimeLimit,
    #                     env_kwargs= {"config_file": "config.json"},
    #                     wrapper_kwargs={"max_episode_steps": 360})
    
    vec_env = unwrap_wrapper(train_env, Monitor) 
    vec_env = VecMonitor(venv=train_env, 
                        filename=f"./logs/{name_prefix}", 
                        info_keywords=('NULL', 'UNLOADED_MODEL', 'LOADED_MODEL', 'invocation_avg', 'action'))  
    
    model = PPO(policy="MultiInputPolicy", n_steps=32, 
                env=train_env, 
                verbose=2, device="cpu",
                tensorboard_log="./ppo_ksim_tensorboard/")
    
    # eval_callback = EvalCallback(eval_env, best_model_save_path="./rl_logs/", n_eval_episodes= 1,
    #                             log_path="./rl_logs/", eval_freq=360,
    #                             deterministic=True, render=False)
        
    checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./logs/",
    name_prefix=name_prefix,
    save_replay_buffer=True,
    save_vecnormalize=True,
    )
    model.learn(total_timesteps=1440*14*100, callback=checkpoint_callback, log_interval=1, progress_bar=True, tb_log_name=name_prefix) 
    model.save(f"./logs/{name_prefix}_ksim_model")

    
