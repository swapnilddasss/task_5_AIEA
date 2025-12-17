import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# Create environment (NO Box2D dependency)
env = gym.make("CartPole-v1")
env = Monitor(env)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

# Train (short run is fine for Task 5)
model.learn(total_timesteps=50_000)

# Get rewards
rewards = env.get_episode_rewards()

# Plot rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO on CartPole-v1")
plt.savefig("reward_curve.png")
plt.show()

env.close()

