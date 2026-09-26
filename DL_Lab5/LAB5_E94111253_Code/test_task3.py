import argparse
import random
from collections import deque

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn


gym.register_envs(ale_py)


class DQN(nn.Module):
    """
    Same CNN architecture used in dqn_task3.py.
    Input: stacked 4 grayscale frames, shape = (4, 84, 84)
    Output: Q-values for each action.
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.network(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class AtariPreprocessor:
    """
    Same preprocessing as the training script:
    RGB frame -> grayscale -> resize to 84x84 -> stack 4 frames.
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # Most likely case: torch.save(model.state_dict(), path)
    if isinstance(checkpoint, dict):
        # If checkpoint is wrapped, try common keys first.
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "q_net_state_dict" in checkpoint:
            checkpoint = checkpoint["q_net_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    return model


def evaluate_one_episode(model, env, preprocessor, device, seed, max_episode_steps=10000, render=False):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state = preprocessor.reset(obs)
    done = False
    total_reward = 0.0
    step_count = 0

    while not done and step_count < max_episode_steps:
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action = model(state_tensor).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = preprocessor.step(next_obs)
        step_count += 1

        if render:
            env.render()

    return total_reward, step_count


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="./results_task3/model_ep1000.pt", help="Path to Task 3 .pt model snapshot.")
    # parser.add_argument("--model_path", type=str, default="./results_task3/LAB5_E94111253_task3_1500000.pt", help="Path to Task 3 .pt model snapshot.")
    parser.add_argument("--model_path", type=str, default="LAB5_E94111253_task3_best.pt", help="Path to Task 3 .pt model snapshot.")
    parser.add_argument("--env_name", type=str, default="ALE/Pong-v5")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--max_episode_steps", type=int, default=10000)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--render", action="store_true", help="Render evaluation. Useful for demo video, slower.")
    args = parser.parse_args()

    render_mode = "human" if args.render else "rgb_array"

    env = gym.make(args.env_name, render_mode=render_mode, frameskip=args.frameskip)
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Environment: {args.env_name}")
    print(f"Model path: {args.model_path}")

    model = DQN(num_actions).to(device)
    model = load_checkpoint(model, args.model_path, device)
    model.eval()

    preprocessor = AtariPreprocessor(frame_stack=4)

    rewards = []
    steps = []

    print("=" * 60)
    print(f"Evaluating seeds {args.start_seed} to {args.start_seed + args.episodes - 1}")
    print("=" * 60)

    for i in range(args.episodes):
        seed = args.start_seed + i
        episode_reward, episode_steps = evaluate_one_episode(
            model=model,
            env=env,
            preprocessor=preprocessor,
            device=device,
            seed=seed,
            max_episode_steps=args.max_episode_steps,
            render=args.render
        )

        rewards.append(episode_reward)
        steps.append(episode_steps)

        print(f"Seed {seed:02d} | Reward: {episode_reward:6.2f} | Steps: {episode_steps}")

    env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes:     {args.episodes}")
    print(f"Mean Reward:  {mean_reward:.2f}")
    print(f"Std Reward:   {std_reward:.2f}")
    print(f"Min Reward:   {min_reward:.2f}")
    print(f"Max Reward:   {max_reward:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

    """
    best_model
        ============================================================
        Evaluation Summary
        ============================================================
        Episodes:     20
        Mean Reward:  18.00
        Std Reward:   1.82
        Min Reward:   15.00
        Max Reward:   21.00
        ============================================================
    """