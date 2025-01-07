import gymnasium as gym
from gymnasium import spaces
import chess
import chess.engine  # For Stockfish integration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pygame
import chess.svg
from io import BytesIO
import cairosvg
import numpy as np
from time import sleep
NGAMES = 2
MOVETIME = 1
reward = 0
prev_eva = 0
counter = 1
prev_reward = 0
EVAL = 0 


class CustomChessEnv(gym.Env):
    def __init__(self):
        super(CustomChessEnv, self).__init__()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-sse41-popcnt.exe")
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 12), dtype=int)

    def reset(self, seed=None, options=None):
        global counter
        super().reset(seed=seed)  # Call the parent class's reset method to handle seeding
        self.board.reset()  # Reset the chess board
        counter = 1
        return self._get_observation(),{}



    def _get_observation(self):
        # Initialize an empty 8x8x12 array for the observation
        observation = np.zeros((8, 8, 12), dtype=int)
        
        # Map pieces to their respective planes in the 12-channel representation
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
            chess.PAWN + 6: 6, chess.KNIGHT + 6: 7, chess.BISHOP + 6: 8, chess.ROOK + 6: 9, chess.QUEEN + 6: 10, chess.KING + 6: 11
        }

        # Populate the observation array based on the current board state
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                # Convert square index to row and column
                row, col = divmod(square, 8)

                # Determine the plane index based on the piece type and color
                piece_type = piece.piece_type
                if piece.color == chess.WHITE:
                    observation[row, col, piece_map[piece_type]] = 1
                else:
                    observation[row, col, piece_map[piece_type + 6]] = 1

        return observation


    def step(self, action):
        global reward, prev_eva, prev_result, counter,prev_reward
        move = self._decode_action(action)

        # Check if the move is legal before applying
        if move not in self.board.legal_moves or move is None:
            info = {"invalid_move": True}  # Add extra info if move is invalid
            #print("illegal move")
            return self._get_observation(), -3, False, False, info  # Penalty for invalid move

        self.board.push(move)  # Push the valid move to the board

        env.render(self,prev_eva*100)  # Print the chessboard
        # Stockfish responds with its move
        stockfish_move = self.engine.play(self.board, chess.engine.Limit(time=0.2)).move
        self.board.push(stockfish_move)
        
        result = self.engine.analyse(self.board, chess.engine.Limit(time=0.2))
        if result["score"].relative.is_mate():
            mate_score = result["score"].relative.mate()
            try:
                eva = 1000.0 / mate_score
            except ZeroDivisionError:
                eva = -1000
        else:
            eva = result["score"].relative.score() / 100  # Scale centipawn scores
        

        env.render(self,eva*100)  # Print the chessboard
        prev_eva = eva
        # Use Stockfish to evaluate the board


        reward = (eva - prev_eva) / 1000.0
        
        prev_result = result
        print(f"{counter}.{move}, {stockfish_move}")
        print(f"evaluation\t=\t{eva:.3f} ")
        print(f"reward\t\t=\t{reward:.3f}")
        print("---------------------------------------")
        counter += 1

        # Game termination
        done = self.board.is_game_over()
        info = {"TimeLimit.truncated": False, "evaluation": eva}  # Add game info
        prev_reward = reward
        if done:
            print("Game Over!")
        # Ensure the environment returns 5 values
            return self._get_observation(), prev_reward , done, False, info  # No truncation, so `False`
        else : 
            return self._get_observation(), reward, done, False, info  # No truncation, so `False`

            
    def _decode_action(self, action):
        legal_moves = list(self.board.legal_moves)
        if 0 <= action < len(legal_moves):
            return legal_moves[action]
        return 0

    def render(self, mode='human', evaluation=None):
        # Generate the SVG representation of the board
        board_svg = chess.svg.board(self.board)

        # Convert SVG to PNG using cairosvg
        board_png = cairosvg.svg2png(bytestring=board_svg.encode('utf-8'))

        # Use BytesIO to load the PNG data into Pygame
        board_image_stream = BytesIO(board_png)

        # Initialize Pygame and create a display window (if not already done)
        pygame.init()
        window_size = 1000  # Window size for the board
        bar_width = 50  # Width of the evaluation bar
        screen = pygame.display.set_mode((window_size + bar_width, window_size))
        pygame.display.set_caption("Chess Board")

        # Load the PNG image from BytesIO
        board_image = pygame.image.load(board_image_stream).convert_alpha()

        # Scale the PNG image to fit the Pygame window
        board_image = pygame.transform.scale(board_image, (window_size, window_size))

        # Display the board
        screen.fill((255, 255, 255))  # Clear screen with white background
        screen.blit(board_image, (0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (window_size, 0, 500, window_size))
        # Normalize evaluation to [-1, 1]
        if evaluation is not None and EVAL :
            # Normalize evaluation value
            max_eval = 1000  # Assuming centipawn range of -400 to +400 for simplicity
            normalized_eval = evaluation / max_eval  # Clamp to [-1, 1]

            # Evaluation bar height and center

            blackh = int(window_size/2)- int(normalized_eval*window_size/4)
            print(blackh)
            # Draw the evaluation bar background
            pygame.draw.rect(screen, (200, 200, 200), (window_size, 0, bar_width, window_size))
            pygame.draw.rect(screen, (0, 0, 0), (window_size, 0, bar_width, blackh))
           


        pygame.display.flip()

        # Keep the board displayed briefly (adjust time if needed)
        pygame.time.wait(MOVETIME)


    def close(self):
        self.engine.quit()


# Initialize the custom environment
env = CustomChessEnv()

# Initialize PPO agent
print("started training")

agent = PPO("MlpPolicy", env, verbose=1)
#agent.learn(total_timesteps=1)  # Adjust the number of timesteps based on your training needs
agent.save("ppo_chess_agent")  # Save the trained model

print("Random gameplay against Stockfish:")
for episode in range(NGAMES):  # Play 5 episodes
    obs_tuple = env.reset()  # `reset()` returns a tuple
    obs = obs_tuple[0]  # Extract the observation only
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)  # Let PPO decide the next action
        obs, reward, done,_, info = env.step(action)  # Execute the action in the environment

env.close()
