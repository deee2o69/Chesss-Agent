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



class CustomChessEnv(gym.Env):
    def __init__(self, display_moves=True, display_game=True):
        super(CustomChessEnv, self).__init__()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-sse41-popcnt.exe")
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 1500})
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 12), dtype=int)  # Board representation
        self.trajectory_buffer = []  # To store the game trajectory
        self.display_moves = display_moves
        self.display_game = display_game
        self.accumulated_reward = 0.0

        

    def reset(self, seed=None, options=None):
        global counter,prev_eva
        super().reset(seed=seed)  # Call the parent class's reset method to handle seeding
        self.board.reset()  # Reset the chess board
        counter = 1
        prev_eva = 0.0
        self.trajectory_buffer = []  # Clear buffer for a new game
        self.accumulated_reward = 0.0  # Reset accumulated reward for the new game

        return self._get_observation()

    def _get_observation(self):
        # Initialize an empty 8x8x12 array for the observation
        observation = np.zeros((8, 8, 12), dtype=int)

        # Map pieces to their respective planes in the 12-channel representation
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
            chess.PAWN + 6: 6, chess.KNIGHT + 6: 7, chess.BISHOP + 6: 8, chess.ROOK + 6: 9, chess.QUEEN + 6: 10, chess.KING + 6: 11
        }

        # Go through each square on the chess board
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                row, col = divmod(square, 8)
                piece_type = piece.piece_type
                if piece.color == chess.WHITE:
                    observation[row, col, piece_map[piece_type]] = 1
                else:
                    observation[row, col, piece_map[piece_type + 6]] = 1

        return observation

    def step(self, action):
        global reward, prev_eva, prev_result, counter 
        move = self._decode_action(action)

        # Check if the move is legal before applying
        if move not in self.board.legal_moves or move is None:
            return self._get_observation(), -3, False, {}, 0  # Penalty for invalid move

        self.board.push(move)  # Push the valid move to the board
        if self.display_game:
            self.render()  # Print the chessboard if enabled
        
        # Stockfish responds with its move
        stockfish_move = self.engine.play(self.board, chess.engine.Limit(time=0.2)).move
        self.board.push(stockfish_move)
        if self.display_game:
            self.render()  # Print the chessboard if enabled

        # Use Stockfish to evaluate the board
        result = self.engine.analyse(self.board, chess.engine.Limit(time=0.2))
        if result['score'].relative.is_mate():
            mate_score = result['score'].relative.mate()
            try:
                eva = 100.0 / mate_score
            except:
                eva = -100

        else:
            eva = result['score'].relative.score() / 100  # Scale centipawn scores to a smaller range
        reward = (eva - prev_eva)
        #print(f'{eva},{prev_eva},{reward}')
        self.accumulated_reward += + reward
        prev_eva = eva

        # Add to trajectory buffer
        self.trajectory_buffer.append({
            "state": self._get_observation().tolist(),  # Convert ndarray to list
            "action": int(action),
            "reward": float(self.accumulated_reward),
            "evaluation": float(eva)
        })

        if self.display_moves:
            print(f"{counter}. {move}, {stockfish_move}")
            print(f"evaluation\t=\t{eva:.3f}")
            print(f"reward\t\t=\t{self.accumulated_reward:.3f}")
            print("---------------------------------------")
        counter += 1

        done = self.board.is_game_over()  # Check if the game is over
        return self._get_observation(), reward, done, {}, eva

    def _decode_action(self, action):
        legal_moves = list(self.board.legal_moves)
        if 0 <= action < len(legal_moves):
            return legal_moves[action]
        return 0

    def render(self, mode='human'):
        board_svg = chess.svg.board(self.board)
        board_png = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))
        board_image_stream = BytesIO(board_png)
        pygame.init()
        window_size = 1000
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Chess Board")
        board_image = pygame.image.load(board_image_stream).convert_alpha()
        board_image = pygame.transform.scale(board_image, (window_size, window_size))
        screen.blit(board_image, (0, 0))
        pygame.display.flip()
        pygame.time.wait(MOVETIME)

    def savegame(self):
        serializable_buffer = []

        # Iterate through each entry in the trajectory buffer
        for entry in self.trajectory_buffer:
            try:
                serializable_entry = {
                    "state": entry["state"].tolist() if isinstance(entry["state"], np.ndarray) else entry["state"],
                    "action": entry["action"],
                    "reward": entry["reward"],
                    "evaluation": entry["evaluation"],
                }
                serializable_buffer.append(serializable_entry)
            except Exception as e:
                print(f"Error serializing entry: {entry}")
                raise e  # Re-raise the exception for debugging

        # Save serialized buffer to a JSON file
        with open("game_trajectories.json", "a") as f:
            json.dump(serializable_buffer, f)
            f.write("\n")

    def close(self):
        self.engine.quit()

# Initialize the custom environment
env = CustomChessEnv(display_moves=False, display_game=True)

# Load saved games
trajectories = load_saved_games("game_trajectories.json")

# Initialize PPO agent
agent = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, gamma=0.99, clip_range=0.2)

# Pretrain the agent
if trajectories:
    print("Pretraining the agent with saved games...")
    pretrain_agent(agent, trajectories)
else:
    print("No saved games found. Starting training from scratch.")

for episode in range(NGAMES):
    obs = env.reset()
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
