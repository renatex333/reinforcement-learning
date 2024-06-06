
from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

tmp_path = "../results/bipedal_walker_a2c/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("BipedalWalker-v3")
vec_env = make_vec_env("BipedalWalker-v3", seed=1, n_envs=1)

model = A2C(
   policy = "MlpPolicy",
   env = vec_env, 
   learning_rate=0.00096, 
   n_steps=8, 
   gamma=0.99, 
   gae_lambda= 0.9,
   vf_coef= 0.4,
   ent_coef= 0.0,
   max_grad_norm= 0.5,
   tensorboard_log=None,
   )

model.set_logger(new_logger)
model.learn(total_timesteps=5_000_000)
model.save("../models/bipedal_walker_a2c")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = A2C.load("../models/bipedal_walker_a2c")

print('modelo treinado')
env = gym.make("BipedalWalker-v3", render_mode='human')
(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()