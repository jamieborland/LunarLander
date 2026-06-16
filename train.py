import gymnasium as gym
import argparse
from utils.logger import Logger
from utils.plotting import plot
import os
import torch
from utils.yaml_import import load_config
from agents.random import RandomAgent
from agents.reinforce import REINFORCE
from agents.dqn import DQN


if __name__ == '__main__':

    env = gym.make("LunarLander-v3")
    episodes = 5000
    best_avg_reward = -float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    parser = argparse.ArgumentParser(description='Train an agent on Lunar Lander')
    parser.add_argument('--agent', type=str, choices=['Random', 'REINFORCE', 'DQN'], required=True)
    args = parser.parse_args()

    AGENTS = {
    'REINFORCE': REINFORCE,
    'DQN': DQN,
}

    if args.agent == 'Random':
        agent = RandomAgent(env.action_space)

    else:
        agent_class = AGENTS[args.agent]
        config = load_config(args.agent)
        agent = agent_class(**config)

    logger = Logger(args.agent)

    for episode in range(episodes):
        episode_rewards = 0
        obs, _ = env.reset() 
        terminated, truncated = False, False
        done = terminated or truncated
        while not done:
            action, log_prob = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, log_prob, reward, next_obs, done)
            agent.update(done)
            episode_rewards += reward
            obs = next_obs

        agent.on_episode_end()
        print(f'Total reward for Episode {episode}/{episodes}: {episode_rewards}')
        logger.log(episode, episode_rewards)

        avg = logger.average_reward()

        if episode >=100 and avg > best_avg_reward:
            best_avg_reward = avg
            agent.save(f"checkpoints/{args.agent}.pth")
            print("Saving model with average reward:", best_avg_reward )
        
        if episode >= 100 and avg > 200:
            print("Environment solved!")
            break

    logger.average_reward()
    print(f'Average reward over last 100 episodes: {logger.average_reward()}')
    logger.save()
    plot(args.agent)





