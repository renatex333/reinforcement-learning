from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

tmp_path = "../results/cartpole_dqn/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("CartPole-v1")
vec_env = make_vec_env("CartPole-v1", n_envs=1)

# docs de toda documentação em https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
#
# model = DQN(
#    policy = "MlpPolicy",
#    env = vec_env, 
#    learning_rate=1e-3, 
#    buffer_size=1_000, 
#    batch_size=64, 
#    gamma=0.99, 
#    exploration_fraction=0.1,
#    exploration_initial_eps=0.9, 
#    exploration_final_eps=0.02, 
#    tau=1,
#    tensorboard_log=None
#    )

model = DQN("MlpPolicy", vec_env, verbose=1)

model.set_logger(new_logger)
model.learn(total_timesteps=300_000)
model.save("../models/cartpole_dqn")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = DQN.load("../models/cartpole_dqn")

print('modelo treinado')
env = gym.make("CartPole-v1", render_mode='human')
(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()