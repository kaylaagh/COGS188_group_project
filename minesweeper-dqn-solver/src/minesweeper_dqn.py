import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from minesweeper import core

class MinesweeperEnv:
    def __init__(self, rows=10, cols=10, mines=30):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.board = core.Board(rows, cols, mines)
    
    def reset(self):
        """
        Resets the environment but keeps the board consistent within the episode.
        """
        random.seed(31)  # Set a fixed seed for consistent board layouts
        np.random.seed(31)
        torch.manual_seed(31)
        
        self.board = core.Board(self.rows, self.cols, self.mines)
        return self.get_state()
    
    def get_state(self):
        """
        Extracts a better representation of the game state.
        """
        board_state = []
        revealed_tiles = 0  # Count how many tiles have been opened
        total_tiles = self.rows * self.cols

        for i in range(self.rows):
            row_state = []
            for j in range(self.cols):
                tile = self.board._tiles[i][j]
                is_revealed = 1 if tile.type != "t" else 0  # 1 if opened, 0 if not
                revealed_tiles += is_revealed

                # Convert tile type into a number representation
                row_state.append(tile.number)
            board_state.append(row_state)

        # Convert board to NumPy array
        board_state = np.array(board_state, dtype=np.float32).flatten()

        # Additional features
        percent_revealed = revealed_tiles / total_tiles  # Percentage of board revealed
        estimated_mines_left = self.board.mines - revealed_tiles  # Estimate remaining mines

        # Return an enriched feature state
        return np.concatenate([board_state, [percent_revealed, estimated_mines_left]])
    
    def step(self, action):
        row, col = divmod(action, self.cols)
        
        if not self.board.tile_valid(row, col):
            return self.get_state(), -5, False, {}  # Penalize invalid moves

        # Check if the tile is already opened
        if self.board._tiles[row][col].type != "t":  # "t" means unopened
            print(f"WARNING: Agent clicked on an already opened tile at ({row}, {col})")
            return self.get_state(), -20, False, {}  # Increased penalty (-20)

        tiles = self.board.tile_open(row, col)  # Open the selected tile
        done = self.board.is_game_over or self.board.is_game_finished

        reward = 0 # Base reward

        if self.board.is_game_over:
            reward = -20  # Losing penalty
        elif self.board.is_game_finished:
            reward = 100  # Winning bonus
        else:
            reward += len(tiles)  # Reward based on number of newly revealed tiles

            # Bonus for revealing a corner
            if (row, col) in [(0, 0), (0, self.cols - 1), (self.rows - 1, 0), (self.rows * self.cols) - 1]:
                reward += 15  

            # Bonus: Reward for opening a tile **near a tile with a "1" bomb
            if self._is_near_one(row, col):
                reward += 10
        

        return self.get_state(), reward, done, {}
    
    def _is_near_one(self, row, col):
        """
        Checks if the given tile (row, col) is adjacent to a tile with a '1'.
        """
        for i in [-1, 0, 1]:  # Check adjacent rows
            for j in [-1, 0, 1]:  # Check adjacent columns
                if i == 0 and j == 0:
                    continue  # Skip the tile itself

                new_row, new_col = row + i, col + j
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    if self.board._tiles[new_row][new_col].type == "1":
                        return True  # Found a '1' nearby

        return False  # No '1' nearby


    
    def get_action_space_size(self):
        return self.rows * self.cols

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, env, lr=0.0003, gamma=0.98, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01, batch_size=64):
        self.env = env
        self.state_size = env.get_state().size
        self.action_size = env.get_action_space_size()
        self.memory = deque(maxlen=50000)  # Increased replay buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.tau = 0.01  # Soft update factor

        # Define policy (online) and target networks
        self.model = QNetwork(self.state_size, self.action_size)  # Online model
        self.target_model = QNetwork(self.state_size, self.action_size)  # Target model

        # Initialize target model with the same weights as the policy model
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Target network is not trained directly

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.
        Avoids already opened tiles.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: Prioritize unexplored tiles
            possible_actions = [
                idx for idx in range(self.action_size)
                if self.env.board._tiles[idx // self.env.cols][idx % self.env.cols].type == "t"
            ]
            
            if possible_actions:
                return random.choice(possible_actions)  # Choose only from unexplored tiles
            return random.randrange(self.action_size)  # If all tiles are open, pick randomly
        
        # Exploitation: Use DQN model
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        # Get the best action from unexplored tiles only
        sorted_actions = q_values.argsort(descending=True).squeeze().tolist()

        for action in sorted_actions:
            row, col = divmod(action, self.env.cols)
            if self.env.board._tiles[row][col].type == "t":  # Check if the tile is unopened
                return action  # Pick the best valid move
        
        return sorted_actions[0]  # Fallback if all tiles are somehow open
    
    def _is_near_one(self, row, col):
        """
        Checks if the given tile (row, col) is adjacent to a tile with a '1'.
        """
        for i in [-1, 0, 1]:  # Check adjacent rows
            for j in [-1, 0, 1]:  # Check adjacent columns
                if i == 0 and j == 0:
                    continue  # Skip the tile itself

                new_row, new_col = row + i, col + j
                if 0 <= new_row < self.env.rows and 0 <= new_col < self.env.cols:
                    if self.env.board._tiles[new_row][new_col].type == "1":
                        return True  # Found a '1' nearby

        return False  # No '1' nearby

    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        next_q_values = self.target_model(next_states).max(1)[0].detach()

        # Compute target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Get current Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Soft update target model
        self.soft_update(self.model, self.target_model, self.tau)


    def soft_update(self, local_model, target_model, tau):
        """
        Perform soft update of target network parameters:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)