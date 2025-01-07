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
from time import sleep
NGAMES = 3
MOVETIME = 1
reward = 0
prev_eva = 0
counter = 1

class CustomChessEnv(gym.Env):
    def __init__(self):
        super(CustomChessEnv, self).__init__()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-sse41-popcnt.exe")
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 12), dtype=int)  # Board representation

    def reset(self, seed=None, options=None):
        global counter
        super().reset(seed=seed)  # Call the parent class's reset method to handle seeding
        self.board.reset()  # Reset the chess board
        counter = 1 
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
                # Calculate the row and column for the square
                row, col = divmod(square, 8)
                # Assign 1 in the respective plane based on the piece type and color
                piece_type = piece.piece_type
                if piece.color == chess.WHITE:
                    observation[row, col, piece_map[piece_type]] = 1
                else:
                    observation[row, col, piece_map[piece_type + 6]] = 1

        return observation


    def step(self, action):
            global reward , prev_eva, prev_result,counter
            move = self._decode_action(action)
            
             # Check if the move is legal before applying
            if move not in self.board.legal_moves or move== None:
                #print(f"Invalid move attempted: {move}")
                return self._get_observation(), -3, False, {},0 # Return a penalty for an invalid move, keep playing

            self.board.push(move)  # Push the valid move to the board
            env.render()  # Print the chessboard
            
            # Stockfish responds with its move
            stockfish_move = self.engine.play(self.board, chess.engine.Limit(time=0.2)).move
            self.board.push(stockfish_move)
            env.render()  # Print the chessboard

            # Use Stockfish to evaluate the board
            result = self.engine.analyse(self.board, chess.engine.Limit(time=0.2))
            if result['score'].relative.is_mate():
                mate_score = result['score'].relative.mate()
                try:
                    eva = 1000.0 / mate_score
                except:
                    eva = -1000

            else:
                eva = result['score'].relative.score() / 100  # Scale centipawn scores to a smaller range
            reward = (eva-prev_eva)/1000.0
            prev_eva = eva
            prev_result = result
            print(f"{counter}.{move},{stockfish_move}")
            print(f"evaluation\t=\t{eva:.3f} ")
            print(f"reward\t\t=\t{reward:.3f}")
            print("---------------------------------------")
            counter +=1
            done = self.board.is_game_over()  # Check if the game is over
            return self._get_observation(), reward, done, {},eva

        
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
        window_size = 1000  # Window size
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
        action, _ = agent.predict(obs, deterministic=False)  # Let PPO decide the next action
        obs, reward, done, info,eva = env.step(action)  # Execute the action in the environment
        if done:
            print(f"Game over.\nReward: {reward}")
            sleep(2)
env.close()
