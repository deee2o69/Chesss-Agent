import gymnasium as gym
from gymnasium import spaces
import chess
import chess.engine
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pygame
import chess.svg
from io import BytesIO
import cairosvg
import numpy as np
from time import sleep,time
import json
import random

NGAMES = 2
MOVETIME = 1
reward = 0.0
prev_eva = 0.0
counter = 1
start_time = time()

def tic_tac():
    """Returns the time elapsed since the start of the program."""
    elapsed_time = time() - start_time
    return elapsed_time

def load_saved_games(file_path="game_trajectories.json"):
    """Load saved game trajectories from a file."""
    trajectories = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                trajectories.append(json.loads(line))
    except FileNotFoundError:
        print(f"No saved games found in {file_path}. Starting fresh.")
    return trajectories


def pretrain_agent(agent, trajectories, batch_size=64):
    """
    Pretrain the PPO agent using saved game trajectories.
    """
    # Flatten the trajectories into a list of transitions
    transitions = []
    for game in trajectories:
        for step in game:
            transitions.append({
                "state": np.array(step["state"]),
                "action": step["action"],
                "reward": step["reward"]
            })
    
    # Shuffle transitions
    random.shuffle(transitions)
    
    # Create batches
    num_batches = len(transitions) // batch_size
    for batch_idx in range(num_batches):
        batch = transitions[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        states = np.array([t["state"] for t in batch])
        actions = np.array([t["action"] for t in batch])
        rewards = np.array([t["reward"] for t in batch])
        
        

        # Now, the agent learns from this batch of accumulated transitions
        agent.learn(total_timesteps=batch_size, reset_num_timesteps=False)

    print(f"Pretrained agent on {len(transitions)} transitions.")





# Initialize the custom environment
env = CustomChessEnv(display_moves=False, display_game=True)

# Load saved games
trajectories = load_saved_games("game_trajectories.json")

# Initialize PPO agent
agent = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, gamma=0.99, clip_range=0.2)

# Pretrain the agent
if trajectories:
    print("Pretraining the agent with saved games...")
    #pretrain_agent(agent, trajectories)
else:
    print("No saved games found. Starting training from scratch.")

for episode in range(NGAMES):
    obs,_ = env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=False)  # Let PPO decide the next action
        obs, reward, done, info, eva = env.step(action)
        if done:
            #print(f"Game over.\nTotal Accumulated Reward: {env.accumulated_reward:.3f}")
            env.savegame()  # Save the game trajectory
            #sleep(2)
print(f"Time elapsed: {tic_tac():.2f} seconds")
env.close()
