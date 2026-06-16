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


    env = gym.make("LunarLander-v3", render_mode="human")
    parser = argparse.ArgumentParser(description='Evaluate an agent on Lunar Lander')
    parser.add_argument('--agent', type=str, choices=['Random', 'REINFORCE', 'DQN'], required=True)
    args = parser.parse_args()

    path = f"checkpoints/{args.agent}.pth"

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

    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.evaluate(obs, path)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated