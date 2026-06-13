import gymnasium as gym


env = gym.make("LunarLander-v3")

print(env.action_space)
print(env.observation_space)
    
