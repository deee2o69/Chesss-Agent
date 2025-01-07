import json
import matplotlib.pyplot as plt
import numpy as np

def load_rewards_from_json(file_path="game_trajectories.json"):
    """Load rewards per step from a JSON file."""
    game_rewards = []  # Store rewards for each game

    try:
        with open(file_path, "r") as f:
            for line in f:
                game_data = json.loads(line)
                game_rewards.append([step["reward"] for step in game_data])
    except FileNotFoundError:
        print(f"No file found at {file_path}. Make sure the game data exists.")
    return game_rewards

def plot_rewards_in_batches(game_rewards_list, batch_size=20, max_moves=40):
    """Plot rewards for each game, grouping every `batch_size` games in a separate figure."""
    num_batches = (len(game_rewards_list) + batch_size - 1) // batch_size  # Calculate the number of batches
    
    for batch_index in range(num_batches):
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, len(game_rewards_list))
        
        plt.figure(figsize=(10, 6))  # Create a new figure for each batch
        
        for game_number, game_rewards in enumerate(game_rewards_list[start:end], start=start + 1):
            # Truncate rewards to the first `max_moves` moves
            truncated_rewards = game_rewards[:max_moves]
            
            # Plot each game's rewards in the current batch
            plt.plot(range(1, len(truncated_rewards) + 1), truncated_rewards, marker='o', linestyle='-', label=f'Game {game_number}')
            
        plt.title(f"Reward Per Move for Games {start + 1} to {end} (First {max_moves} Moves)")
        plt.xlabel("Move Number")
        plt.ylabel("Reward")
        plt.grid(True)
        #plt.legend()
        plt.show()

if __name__ == "__main__":
    # Load rewards for each move in each game from the JSON file
    game_rewards_list = load_rewards_from_json("game_trajectories.json")

    if game_rewards_list:
        # Plot rewards for all games, grouped every 100 games in separate figures
        plot_rewards_in_batches(game_rewards_list, max_moves=40)
    else:
        print("No rewards to plot. Play some games first!")
