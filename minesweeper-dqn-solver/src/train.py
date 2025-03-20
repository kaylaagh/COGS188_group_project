import torch
import matplotlib.pyplot as plt
from minesweeper_dqn import MinesweeperEnv, DQNAgent

def train_agent(num_episodes=1000, max_steps=100):
    env = MinesweeperEnv(rows=10, cols=10, mines=30)
    agent = DQNAgent(env)
    
    episode_rewards = []
    win_rates = []
    episode_lengths = []  # Track how long the agent survives
    exploration_rates = []  # Track how epsilon decays
    tiles_revealed = []  # Track how many tiles the agent reveals per episode
    win_count = 0  # Track number of successful games

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        # Count how many tiles are revealed at the start of the episode
        initial_revealed = sum(1 for row in env.board._tiles for tile in row if tile.type != "t")

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1  # Count the steps taken

            if done:
                # Check if the agent won the game
                if env.board.is_game_finished:
                    win_count += 1
                break
        
        # Count how many tiles are revealed at the end of the episode
        final_revealed = sum(1 for row in env.board._tiles for tile in row if tile.type != "t")
        revealed_tiles = final_revealed - initial_revealed  # Only count newly revealed tiles

        # Store performance metrics
        episode_rewards.append(total_reward)
        win_rates.append(win_count / (episode + 1) * 100)  # Update win rate
        episode_lengths.append(steps)  # Store episode length
        exploration_rates.append(agent.epsilon)  # Store epsilon decay
        tiles_revealed.append(revealed_tiles)  # Store number of revealed tiles in this episode

        print(f"Episode {episode+1}/{num_episodes} - Reward: {total_reward:.2f}, Win Rate: {win_rates[-1]:.3f}%, Steps: {steps}, Tiles Revealed: {revealed_tiles}, Epsilon: {agent.epsilon:.3f}")

    return episode_rewards, win_rates, episode_lengths, exploration_rates, tiles_revealed
