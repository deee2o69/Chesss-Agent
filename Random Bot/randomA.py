import gymnasium as gym
from gymnasium import spaces
import chess
import chess.engine  # For Stockfish integration
from stable_baselines3 import PPO
import pygame
import chess.svg
from io import BytesIO
import cairosvg

NGAMES = 1
MOVETIME = 2000

class CustomChessEnv(gym.Env):
    def __init__(self):
        super(CustomChessEnv, self).__init__()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-sse41-popcnt.exe")
        self.action_space = spaces.Discrete(64 * 64)  # 64 squares * 64 possible target squares
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 12), dtype=int)  # Board representation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Call the parent class's reset method to handle seeding
        self.board.reset()  # Reset the chess board
        return self._get_observation()

    def _get_observation(self):
        # Convert board state to a numerical representation (e.g., bitboard or array)
        return None

    def step(self, action):
        try:
            move = self._decode_action(action)
            
            # Retry until a legal move is generated
            while move not in self.board.legal_moves:
                action = self.action_space.sample()  # Generate a new random action
                move = self._decode_action(action)
            
            self.board.push(move)  # Push the valid move to the board

            # Stockfish responds with its move
            stockfish_move = self.engine.play(self.board, chess.engine.Limit(time=0.1)).move
            self.board.push(stockfish_move)

            # Use Stockfish to evaluate the board
            result = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
            reward = result['score'].relative.score() if result['score'].relative else 0

            done = self.board.is_game_over()  # Check if the game is over
            return self._get_observation(), reward, done, {}

        except Exception as e:
            print(f"Error in step: {e}")
            return self._get_observation(), -1, True, {}
    def _decode_action(self, action):
        # Example: Map action index to UCI moves
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)
        return move if move in self.board.legal_moves else None

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

# Initialize PPO agent
agent = PPO("MlpPolicy", env, verbose=1)

print("Random gameplay against Stockfish:")
for episode in range(NGAMES):  # Play 5 episodes
    obs = env.reset()
    done = False
    while not done:
        # Random action
        random_action = env.action_space.sample()
        obs, reward, done, info = env.step(random_action)
        env.render()  # Print the chessboard
        if done:
            print(f"Game over. Reward: {reward}")
env.close()
