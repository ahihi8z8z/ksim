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

name_prefix = "PPO"


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

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

if __name__=="__main__":
    train_env = make_vec_env(env_id="KsimEnv-V0", n_envs=12, 
                        vec_env_cls=SubprocVecEnv, 
                        env_kwargs= {"config_file": "config.json",
                                    "render_dir": f"logs/{name_prefix}"})

    train_vec_env = VecMonitor(venv=train_env, 
                        filename=f"logs/{name_prefix}")  
    
    eval_env = make_vec_env(env_id="KsimEnv-V0", n_envs=1, 
                    vec_env_cls=SubprocVecEnv, 
                    env_kwargs= {"config_file": "config.json",
                                "render_dir": f"logs/{name_prefix}"})

    val_vec_env = VecMonitor(venv=eval_env, 
                        filename=f"logs/{name_prefix}")  
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{name_prefix}",
                                log_path=f"./logs/{name_prefix}", eval_freq=1440/2, n_eval_episodes = 1,
                                deterministic=True, render=False)

    # checkpoint_callback = SaveOnBestTrainingRewardCallback(log_dir=f"logs/{name_prefix}",
    #                                                       check_freq=144)
    model = PPO(policy="MultiInputPolicy", n_steps=32, 
                env=train_vec_env, 
                verbose=2, device="cpu",
                tensorboard_log="./ksim_tensorboard/")
    
    model.learn(total_timesteps=1440*14*30, 
                callback=eval_callback, 
                log_interval=1, progress_bar=True, 
                tb_log_name=name_prefix) 
    
    model.save(f"./logs/{name_prefix}/ksim_model")
           
