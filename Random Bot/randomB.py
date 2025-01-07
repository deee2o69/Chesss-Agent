import gymnasium as gym
from gymnasium import spaces
import chess
import chess.engine  # For Stockfish integration
from stable_baselines3 import PPO
import pygame
import chess.svg
from io import BytesIO
import cairosvg
import numpy as np

NGAMES = 10
MOVETIME = 500

class CustomChessEnv(gym.Env):
    def __init__(self):
        super(CustomChessEnv, self).__init__()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-sse41-popcnt.exe")
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 12), dtype=int)  # Board representation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Optional, for reproducibility
        self.board.reset()  # Reset the chessboard
        return self._get_observation()  # Return the board state as the only output

    def _get_observation(self):
        # Create an 8x8x12 array to represent the board state
        board_state = np.zeros((8, 8, 12), dtype=int)

        # Map chess piece types to indices (0-5 for White, 6-11 for Black)
        piece_value_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

        # Iterate through all squares on the board
        for square, piece in self.board.piece_map().items():
            # Ensure square is an integer index (0–63)
            if isinstance(square, int):
                row, col = divmod(square, 8)  # Convert square index to (row, col)
                if piece.piece_type in piece_value_map:
                    layer = piece_value_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
                    if 0 <= layer < 12:  # Ensure layer is within bounds
                        board_state[row, col, layer] = 1

        return board_state
   
    def step(self, action):
        try:
            move = self._decode_action(action)
            if move is None or not self.board.is_legal(move):
                # End the game immediately for invalid or illegal moves
                return self._get_observation(), 0, True, {"error": "illegal move"}
            
            self.board.push(move)  # Push the valid move to the board
            env.render()  # Display the chessboard

            # Stockfish responds with its move
            stockfish_move = self.engine.play(self.board, chess.engine.Limit(time=0.1)).move
            self.board.push(stockfish_move)

            # Use Stockfish to evaluate the board
            result = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
            reward = result['score'].relative.score() if result['score'].relative else 0
            try:
                reward = reward/100
            except:
                None
            done = self.board.is_game_over()  # Check if the game is over
            return self._get_observation(), reward, done, {}

        except Exception as e:
            print(f"Error in step: {e}")
            return self._get_observation(), -1, True, {}
        
    def _decode_action(self, action):
        legal_moves = list(self.board.legal_moves)
        if 0 <= action < len(legal_moves):
            return legal_moves[action]
        return 0

    def render(self, mode='human'):
        # Generate the SVG representation of the board
        board_svg = chess.svg.board(self.board)

        # Convert SVG to PNG using cairosvg
        board_png = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))

        # Use BytesIO to load the PNG data into Pygame
        board_image_stream = BytesIO(board_png)

        # Initialize Pygame and create a display window (if not already done)
        pygame.init()
        window_size = 600  # Window size
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Chess Board")

        # Load the PNG image from BytesIO
        board_image = pygame.image.load(board_image_stream).convert_alpha()

        # Scale the PNG image to fit the Pygame window
        board_image = pygame.transform.scale(board_image, (window_size, window_size))

        # Display the board
        screen.blit(board_image, (0, 0))
        pygame.display.flip()

        # Keep the board displayed briefly (adjust time if needed)
        pygame.time.wait(MOVETIME)
        
    def close(self):
        self.engine.quit()


# Initialize the custom environment
env = CustomChessEnv()
obs = env.reset()
#print(obs.shape)  # Should print (8, 8, 12)
#print(obs)  # Verify the board state representation
# Initialize PPO agent
agent = PPO("MlpPolicy", env, verbose=1)
#agent.learn(total_timesteps=100)

print("Gameplay against Stockfish:")
for episode in range(NGAMES):  # Play 1 episode
    obs = env.reset()
    done = False
    while not done:
        # Get action from the agent
        action, _ = agent.predict(obs, deterministic=True)  # Unpack the action tuple
        obs, reward, done, _ = env.step(action)  # Pass the action to the environment
        print("Reward:", reward)
        env.render()  # Display the chessboard
        if done:
            print(f"Game over. Reward: {reward}")

env.close()
