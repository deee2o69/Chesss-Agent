import json
import matplotlib.pyplot as plt
import numpy as np

def load_rewards_from_json(file_path="game_trajectories.json"):
    """Load rewards per game from a JSON file."""
    rewards_per_game = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                game_data = json.loads(line)
                total_reward = sum(step["reward"] for step in game_data)
                rewards_per_game.append(total_reward)
    except FileNotFoundError:
        print(f"No file found at {file_path}. Make sure the game data exists.")
    return rewards_per_game

def plot_rewards(rewards):
    """Plot the rewards per game over time with a mean slope line."""
    plt.figure(figsize=(10, 6))

    # Plot the rewards per game
    plt.plot(range(1, len(rewards) + 1), rewards, marker='o', linestyle='-', color='b', label='Reward')

    # Calculate and plot the mean slope line
    slopes = []
    for i in range(1, len(rewards) + 1):
        mean_slope = np.mean(rewards[:i])
        slopes.append(mean_slope)
    plt.plot(range(1, len(slopes) + 1), slopes, linestyle='--', color='r', label='Mean Slope')

    plt.title("Reward Per Game Over Time")
    plt.xlabel("Game Number")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load rewards from the JSON file
    rewards = load_rewards_from_json("game_trajectories.json")

    if rewards:
        # Plot the rewards
        plot_rewards(rewards)
    else:
        print("No rewards to plot. Play some games first!")
