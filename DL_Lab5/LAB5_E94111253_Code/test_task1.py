import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)


def evaluate(model_path, episodes=20, render=False):
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rewards = []

    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0

        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state).argmax(dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {seed}: Reward = {total_reward}")

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    task1_score_percent = min(avg_reward, 480) / 480 * 15

    print("=" * 50)
    print(f"Rewards: {rewards}")
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"Estimated Task 1 Score: {task1_score_percent:.2f} / 15")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LAB5_E94111253_task1.pt")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    evaluate(args.model_path, args.episodes)

    """
    best_model.pt   |   Rewards: [431.0, 279.0, 500.0, 500.0, 191.0, 214.0, 500.0, 200.0, 268.0, 218.0, 212.0, 500.0, 500.0, 188.0, 198.0, 302.0, 500.0, 263.0, 500.0, 500.0]
                    |   Average Reward over 20 episodes: 348.20
                    |   Std Reward: 134.21
                    |   Estimated Task 1 Score: 10.88 / 15
    model_ep1900    |   Rewards: [500.0, 500.0, 500.0, 500.0, 292.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 454.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
                    |   Average Reward over 20 episodes: 487.30
                    |   Std Reward: 45.91
                    |   Estimated Task 1 Score: 15.00 / 15
    model_ep1800    |   Rewards: [500.0, 500.0, 500.0, 500.0, 292.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
                    |   Average Reward over 20 episodes: 489.60
                    |   Std Reward: 45.33
                    |   Estimated Task 1 Score: 15.00 / 15
    model_ep1700    |   Rewards: [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
                    |   Average Reward over 20 episodes: 500.00
                    |   Std Reward: 0.00
                    |   Estimated Task 1 Score: 15.00 / 15
    
    """