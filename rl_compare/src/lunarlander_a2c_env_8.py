from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

tmp_path = "../results/lunarlander_a2c_env_8/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = make_vec_env("LunarLander-v2", n_envs=8)
model = A2C(
   policy = "MlpPolicy",
   env = env, 
   learning_rate=1e-2, 
   gamma=0.99, 
   tensorboard_log=None
   )

model.set_logger(new_logger)
model.learn(total_timesteps=100_000)
model.save("../models/lunarlander_a2c_env_8")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = A2C.load("../models/lunarlander_a2c_env_8")

print('modelo treinado')
env = gym.make("LunarLander-v2", render_mode='human')
(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:  
      (obs,_) = env.reset()
