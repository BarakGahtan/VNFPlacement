import subprocess
import time
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from rl_folder.env import FunctionPlacementEnv
from rl_folder.utils import TensorboardCallback
from utils1 import input_parser
import optuna
from torch.utils.tensorboard import SummaryWriter


class CustomRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomRewardCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir='./logs/training')

    def _on_step(self) -> bool:
        if self.n_calls % self.model.n_steps == 0:
            rewards = self.locals['rewards']
            self.writer.add_scalar('train/mean_reward', np.mean(rewards), self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()


def evaluate_model(env, model, n_episodes=10):
    total_rewards = []
    for _ in range(n_episodes):
        result = env.reset()
        if isinstance(result, tuple):
            obs, _ = result  # Handle the returned tuple
        else:
            obs = result
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, _ = step_result
            else:
                obs, reward, done, _ = step_result
                truncated = False
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def objective(trial, env_params, n_episodes=100, n_steps=100):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048]) #
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)

    # Create the environment
    env = FunctionPlacementEnv(**env_params, reward_params={'overload_penalty': 1, 'variance_penalty': 1})  # Default reward params
    env = Monitor(env)  # Wrap the environment with Monitor
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Define evaluation environment and callback
    eval_env = DummyVecEnv([lambda: Monitor(FunctionPlacementEnv(**env_params, reward_params={'overload_penalty': 25, 'variance_penalty': 1}))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model/',
                                 log_path='./logs/', eval_freq=500,
                                 deterministic=True, render=False)

    # Initialize the PPO model with suggested hyperparameters
    model = PPO('MlpPolicy', env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                verbose=1, tensorboard_log='./logs/')

    # Start TensorBoard
    port = 6006 + trial.number
    tensorboard_command = f"tensorboard --logdir=./logs/ --host=127.0.0.1 --port={port}"
    tensorboard_process = subprocess.Popen(tensorboard_command, shell=True)
    time.sleep(5)  # Ensure TensorBoard starts

    # Train the model with custom reward callback
    custom_reward_callback = CustomRewardCallback()
    model.learn(total_timesteps=n_episodes * n_steps, callback=[eval_callback, custom_reward_callback])

    # Evaluate the model
    mean_reward, std_reward = evaluate_model(env, model, n_episodes=10)

    # Terminate TensorBoard process
    tensorboard_process.terminate()

    return mean_reward


if __name__ == "__main__":
    parsed_args = input_parser.Parser()
    opts = parsed_args.parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env_params = {
        'num_servers': opts.servers_cnt,
        'num_functions': opts.possible_func,
        'subset_functions': opts.demanded_func,
        'num_clients': opts.clients_cnt,
        'params': opts
    }

    # Perform hyperparameter optimization using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, env_params, n_episodes=opts.episode_count, n_steps=opts.steps_count), n_trials=20)

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")

    # Use the best hyperparameters found by Optuna for further training or testing
    best_hyperparams = study.best_trial.params

    # Visualization
    df = pd.DataFrame(study.trials_dataframe())

    plt.figure(figsize=(10, 8))
    plt.plot(df['number'], df['value'])
    plt.xlabel('Trial Number')
    plt.ylabel('Mean Reward')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig('optuna_optimization_results.png')
    plt.show()

    # Open TensorBoard for the final log directory used
    final_log_dir = f"./logs/best_model/"
    tensorboard_command = f"tensorboard --logdir={final_log_dir} --host=127.0.0.1 --port=6006"
    final_tensorboard_process = subprocess.Popen(tensorboard_command, shell=True)
    print(f"TensorBoard is running at http://127.0.0.1:6006")

    # Wait for user input to close TensorBoard
    input("Press Enter to close TensorBoard...")
    final_tensorboard_process.terminate()
