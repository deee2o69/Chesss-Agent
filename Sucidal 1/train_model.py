import gymnasium as gym
from gymnasium import spaces
import chess
import chess.engine
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Using DummyVecEnv
import pygame
import chess.svg
from io import BytesIO
import cairosvg
import numpy as np
from time import sleep, time
import json
import random
import os
from chess_env import CustomChessEnv  # Import the custom chess environment
import torch.nn as nn


def train_ppo_agent(model_save_path):
    """
    Train a PPO agent using the CustomChessEnv and save the model.

    Args:
        model_save_path (str): Path to save the trained PPO model.
    """
    # Create a DummyVecEnv with the CustomChessEnv to ensure single-core execution
    envs = DummyVecEnv([lambda: CustomChessEnv(display_moves=False, display_game=False)])  # Wrapping in DummyVecEnv
    
    # Initialize the PPO model with an MLP policy
    if os.path.exists(model_save_path + ".zip"):
        print("Loading pre-trained model...")
        # Load the pre-trained model
        model = PPO.load(model_save_path, env=envs)  # Ensure the environment is passed when loading
    else:
        print("Creating a new model...")
        # Create a new model if it doesn't exist
        # Initialize PPO model using MlpPolicy for the custom Box observation space
        model = PPO(
            "MlpPolicy",                # Use MlpPolicy for the Box observation space
            envs,                        # The environment (now wrapped in DummyVecEnv for single-core)
            verbose=1,
            learning_rate=0.0003,
            n_steps=4096,
            batch_size=512,
            ent_coef=0.01,
            clip_range=0.2,
            tensorboard_log="./ppo_chess_tensorboard",
            n_epochs=10,
            policy_kwargs=dict(
                net_arch=[128, 128, 64],  # Fully connected layers (you can adjust this)
                activation_fn=nn.ReLU  # Pass the ReLU function instead of a string
            ),
            gae_lambda=0.95,
            gamma=0.99,
        )

    print("Training PPO agent...")

    # Loop while checking the elapsed time
    x = 2**16  # You can adjust the number of timesteps here
    print(x)
    for _ in range(16):
        print("Loading pre-trained model...")
        model = PPO.load(model_save_path, env=envs)  # Ensure the environment is passed when loading
        model.learn(total_timesteps=x)
        print("Saving PPO agent...")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    
if __name__ == "__main__":
    model_save_path = "ppo_chess_agent"

    # Train and save the PPO agent
    train_ppo_agent(model_save_path)
