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

MOVETIME = 1000
NGAMES = 10


class CustomChessEnv(gym.Env):
    def __init__(self):
        super(CustomChessEnv, self).__init__()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-sse41-popcnt.exe")
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 12), dtype=int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))
        return self._get_observation()

    def _get_observation(self):
        board_state = np.zeros((8, 8, 12), dtype=int)
        piece_value_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        for square, piece in self.board.piece_map().items():
            row, col = divmod(square, 8)
            layer = piece_value_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            board_state[row, col, layer] = 1
        return board_state

    def _decode_action(self, action):
        legal_moves = list(self.board.legal_moves)
        if 0 <= action < len(legal_moves):
            return legal_moves[action]
        return 0

    def step(self, action):
        try:
            # Decode the action to a move
            move = self._decode_action(action)
            if move is None or not self.board.is_legal(move):
                # End the game immediately for invalid or illegal moves
                return self._get_observation(), 0, True, {"error": "illegal move"}

            # Apply the player's move
            self.board.push(move)

            # Check if the player's move ends the game
            if self.board.is_checkmate():
                return self._get_observation(), 0, True, {"result": "checkmate"}
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                return self._get_observation(), 0, True, {"result": "draw"}

            # Stockfish's turn
            stockfish_move = self.engine.play(self.board, chess.engine.Limit(time=0.1)).move
            self.board.push(stockfish_move)

            # Check if Stockfish's move ends the game
            if self.board.is_checkmate():
                return self._get_observation(), 0, True, {"result": "stockfish_checkmate"}
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                return self._get_observation(), 0, True, {"result": "draw"}

            # Update the action space for the next step
            self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))

            # Continue the game
            return self._get_observation(), 0, False, {}
        except Exception as e:
            print(f"Error in step: {e}")
            return self._get_observation(), 0, True, {"error": str(e)}




    def render(self, mode='human'):
        board_svg = chess.svg.board(self.board)
        board_png = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))
        board_image_stream = BytesIO(board_png)

        pygame.init()
        window_size = 600
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Chess Board")

        board_image = pygame.image.load(board_image_stream).convert_alpha()
        board_image = pygame.transform.scale(board_image, (window_size, window_size))

        screen.blit(board_image, (0, 0))
        pygame.display.flip()
        pygame.time.wait(MOVETIME)

    def close(self):
        self.engine.quit()


# Initialize the environment
env = CustomChessEnv()
obs = env.reset()

# Initialize PPO agent
agent = PPO("MlpPolicy", env, verbose=1)

# Train the agent
#agent.learn(total_timesteps=10000)

# Play games
print("Gameplay against Stockfish:")
for episode in range(NGAMES):
    obs = env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=False)  # Use non-deterministic policy for exploration
        obs, reward, done, _ = env.step(action)
        #print("Reward:", reward)
        #print("Evaluation :", evaluation / 100)
        env.render()
        if done:
            print(f"Game over. Reward: {reward}")

env.close()
