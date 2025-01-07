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


class CustomChessEnv(gym.Env):
    def __init__(self, display_moves=False, display_game=False,Move_time=0.01):
        super(CustomChessEnv, self).__init__()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-sse41-popcnt.exe")
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 3000})
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))
        self.observation_space = spaces.Box(0, 1, shape=(8, 8, 12), dtype=int)  # Board representation
        self.trajectory_buffer = []  # To store the game trajectory
        self.display_moves = display_moves
        self.display_game = display_game
        self.reward = 0.0
        self.prev_eva =0.0
        self.counter = 1
        self.accumulated_reward = 0.0
        self.MOVETIME = Move_time
        

    def reset(self, seed=None, options=None):
        global counter,prev_eva
        super().reset(seed=seed)  # Call the parent class's reset method to handle seeding
        self.board.reset()  # Reset the chess board
        self.counter = 1
        self.prev_eva = 0.0
        self.trajectory_buffer = []  # Clear buffer for a new game
        self.accumulated_reward = 0.0  # Reset accumulated reward for the new game

        return self._get_observation(),{}

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
        #print(action)
        move = self._decode_action(action)
        obser = self._get_observation() 
        # Check if the move is legal before applying
        if move not in self.board.legal_moves or move is None:
            self.accumulated_reward-=0.5
            self.trajectory_buffer.append({
            "state": obser.tolist(),  # Convert ndarray to list
            "action": int(action),
            "reward": float(-0.5),
            "evaluation": float(self.prev_eva)
        })
            
            return self._get_observation(), self.reward-0.5, False,False, {}  # Penalty for invalid move

        self.board.push(move)  # Push the valid move to the board
        if self.display_game:
            self.render()  # Print the chessboard if enabled
        
        # Stockfish responds with its move
        stockfish_move = self.engine.play(self.board, chess.engine.Limit(time=self.MOVETIME)).move
        self.board.push(stockfish_move)
        if self.display_game:
            self.render()  # Print the chessboard if enabled

        # Use Stockfish to evaluate the board
        result = self.engine.analyse(self.board, chess.engine.Limit(time=self.MOVETIME))
        if result['score'].relative.is_mate():
            mate_score = result['score'].relative.mate()
            if mate_score > 0:
                eva = self.prev_eva + 10.0/mate_score
                self.reward = eva 
            elif mate_score < 0:
                eva = self.prev_eva + 10.0/mate_score
                self.reward = eva
            else :
                eva = self.prev_eva
                
            if self.board.is_game_over():
                if self.board.result() == "1-0":  # Bot wins
                    self.reward = self.prev_eva +5.0
                elif self.board.result() == "0-1":  # Bot loses
                    self.reward = self.prev_eva -5.0  # Heavy penalty
                else:  # Draw
                    self.reward = self.prev_eva + 1.0  # Small positive reward
                
        else:
            eva = result['score'].relative.score() / 100  # Scale centipawn scores to a smaller range
            self.reward = eva - self.prev_eva
            
        self.reward += (self.counter / 100.0) * 0.1 
        if self.reward > 0:
            self.reward *= (2 + min(1.0, self.counter / 100.0))
        else :
            self.reward *= 0.1
        self.accumulated_reward += self.reward
        self.prev_eva = eva

        # Add to trajectory buffer
        self.trajectory_buffer.append({
            "state": obser.tolist(),  # Convert ndarray to list
            "action": int(action),
            "reward": float(self.reward),
            "evaluation": float(self.prev_eva)
        })

        if self.display_moves:
            print(f"{self.counter}. {move}, {stockfish_move}")
            print(f"evaluation\t=\t{eva:.3f}")
            print(f"reward\t\t=\t{self.reward:.3f}")
            print("---------------------------------------")
        self.counter += 1

        done = self.board.is_game_over()  # Check if the game is over
        return self._get_observation(), self.reward, done,False, {}

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
        sleep(self.MOVETIME)

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
