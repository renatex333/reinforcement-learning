from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

tmp_path = "../results/lunar_lander_ppo_env_1/"

new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("LunarLander-v2")
vec_env = make_vec_env("LunarLander-v2", n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=100_000)
model.save("../models/lunar_lander_ppo_env_1")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = PPO.load("../models/lunar_lander_ppo_env_1")

env = gym.make("LunarLander-v2", render_mode='human')
(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()