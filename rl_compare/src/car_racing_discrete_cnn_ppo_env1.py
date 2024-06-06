from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

tmp_path = "../results/car_racing_discrete_cnn_env1"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("CarRacing-v2")
vec_env = make_vec_env("CarRacing-v2", n_envs=1)

# docs de toda documentação em https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
#
model = PPO(
  policy = "CnnPolicy", 
  env = vec_env,
  learning_rate=1e-3,
  n_steps=512,
  batch_size=128,
  n_epochs=10,
  gamma=0.99,
  gae_lambda=0.95,
  clip_range=0.2,
  vf_coef=0.5,
  ent_coef=0.0,
  max_grad_norm=0.5,
  tensorboard_log=None
)

model.set_logger(new_logger)
model.learn(total_timesteps=600_000)
model.save("../models/car_racing_ppo_cnn_env1")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = PPO.load("../models/car_racing_ppo_cnn_env1")

print('modelo treinado')
env = gym.make("CarRacing-v2", render_mode='human')
(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()