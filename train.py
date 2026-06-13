import gymnasium as gym
import argparse
from utils.logger import Logger
from utils.plotting import plot

env = gym.make("LunarLander-v3")

episodes = 1000

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train an agent on Lunar Lander')
    parser.add_argument('--model', type=str, choices=['Random'], required=True)
    args = parser.parse_args()

    if args.model == 'Random':
        from agents.random import RandomAgent
        agent = RandomAgent(env.action_space)
        logger = Logger(args.model)
    
    for episode in range(episodes):
        episode_rewards = 0
        obs, _ = env.reset() 
        terminated, truncated = False, False
        done = terminated or truncated
        while not done:
            action, log_prob = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, next_obs, done)
            agent.update()
            episode_rewards += reward
            obs = next_obs
        
        print(f'Total reward for Episode {episode}/{episodes}: {episode_rewards}')
        logger.log(episode, episode_rewards)


    logger.average_reward()
    print(f'Average reward over last 100 episodes: {logger.average_reward()}')
    logger.save()
    plot(args.model)





