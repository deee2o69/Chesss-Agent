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
from time import sleep, time
import json
import random
from chess_env import CustomChessEnv

NGAMES = 10
start_time = time()

def tic_tac():
    global start_time 
    """Returns the time elapsed since the start of the program."""
    elapsed_time = time() - start_time
    start_time = time()
    return elapsed_time

# Initialize the custom environment
env = CustomChessEnv(display_moves=True, display_game=True,Move_time = 0.2)  

# Load the pre-trained PPO agent
model_save_path = "ppo_chess_agent"
agent = PPO.load(model_save_path, env=env)

# Play games using the trained agent
for episode in range(NGAMES):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=False)  # Use deterministic=True for consistent actions
        obs, reward, done, truncated, info = env.step(action)
        
        # After each move, render the board
        env.render()
        
        if done:
            print(f"Game {episode+1}, accumulated_reward: {env.accumulated_reward:.3f}, Time: {tic_tac():.2f}s")
            env.savegame()  # Save the game trajectory

env.close()
