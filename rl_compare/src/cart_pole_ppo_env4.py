
from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

tmp_path = "../results/cartpole_ppo_env4/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("CartPole-v1")
vec_env = make_vec_env("CartPole-v1", n_envs=4)

# docs de toda documentação em https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
#
# model = PPO(
#    policy = "MlpPolicy",
#    env = vec_env, 
#    learning_rate=1e-3, 
#    batch_size=64, 
#    gamma=0.99
#    )

model = PPO(
   policy = "MlpPolicy",
   env = vec_env)

model.set_logger(new_logger)
model.learn(total_timesteps=300_000)
model.save("../models/cartpole_ppo_env4")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = PPO.load("../models/cartpole_ppo_env4")

print('modelo treinado')
env = gym.make("CartPole-v1", render_mode='human')
(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()