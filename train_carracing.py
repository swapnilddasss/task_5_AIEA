import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

model.learn(total_timesteps=50_000)
rewards = env.get_episode_rewards()


plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO on CartPole-v1")
plt.savefig("reward_curve.png")
plt.show()

env.close()

