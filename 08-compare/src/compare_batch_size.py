from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import Process

def execute_training(d):
    tmp_path = f"../results/compare_batch_size_{d}/"
    new_logger = configure(tmp_path, ["csv"])

    env = gym.make("LunarLander-v2")
    vec_env = make_vec_env("LunarLander-v2", n_envs=1)

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        replay_buffer_class=100,
        batch_size=d,
        tensorboard_log=None,
    )

    model.set_logger(new_logger)
    model.learn(total_timesteps=500_000)
    model.save(f"../models/compare_batch_size_{d}")

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Batch size = {d}, Mean reward: {mean_reward} +/- {std_reward:.2f}")


D = [32, 64, 128, 256, 512]
for d in D:
    process = Process(target=execute_training, args=(d,))
    process.start()