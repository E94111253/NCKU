# Value-Based Reinforcement Learning with DQN

This project implements Deep Q-Network (DQN) agents for **CartPole-v1** and **ALE/Pong-v5**. It compares a basic DQN with an enhanced Pong agent using Double DQN, prioritized experience replay (PER), and 3-step returns. The accompanying lab report is `LAB5_E94111253.pdf` (add it to the repository root if you want the report available on GitHub).

## Tasks

| Task | Environment | Observation / network | Replay | Target and return |
| --- | --- | --- | --- | --- |
| 1 | CartPole-v1 | 4-dimensional state / MLP | Uniform | DQN, 1-step |
| 2 | ALE/Pong-v5 | Four stacked preprocessed frames / CNN | Uniform | DQN, 1-step |
| 3 | ALE/Pong-v5 | Four stacked preprocessed frames / CNN | Prioritized | Double DQN, 3-step |

## Repository contents

| File | Purpose |
| --- | --- |
| `dqn_task1.py`, `dqn_task2.py`, `dqn_task3.py` | Training scripts for Tasks 1–3 |
| `test_task1.py`, `test_task2.py`, `test_task3.py` | Evaluate saved model weights |
| `requirements.txt` | Python dependencies for training (see evaluation note below) |

The provided code archive contains **no pretrained `.pt` weights**. Run training first, or supply your own compatible checkpoints to evaluate the agents.

## Setup

From the directory containing the Python files:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For `test_task2.py`, also install `imageio` and an FFmpeg backend because the script writes an MP4 for every evaluation episode:

```bash
python -m pip install imageio imageio-ffmpeg
```

The training scripts log to Weights & Biases (`wandb.init`). Set up your W&B account/login before training, or set `WANDB_MODE=offline` in your shell to keep logs locally. Pong training is computationally demanding; a CUDA GPU is helpful.

## Train

```bash
python dqn_task1.py
python dqn_task2.py
python dqn_task3.py
```

Override the checkpoint destination with `--save-dir`, for example:

```bash
python dqn_task3.py --save-dir results_task3
```

The default output folders are `results_task1/`, `results_task2/`, and `results_task3_600k/`. Training periodically writes `model_ep*.pt` and may write `best_model.pt` when evaluation reward improves. Task 3 also writes step-based `task3_*.pt` snapshots. The saved files are PyTorch model state dictionaries.

## Evaluate

Pass a checkpoint path explicitly; the test scripts' default filenames are **not included** in the code archive and differ from the training output paths.

```bash
python test_task1.py --model-path results_task1/model_ep1900.pt --episodes 20
python test_task2.py --model-path results_task2/best_model.pt --episodes 20 --output-dir eval_videos
python test_task3.py --model_path results_task3_600k/best_model.pt --episodes 20
```

Replace each example path with a checkpoint that actually exists after your run. Task 2 saves episode videos under `eval_videos/`; Task 3 accepts `--render` to show the game during evaluation. All three test scripts select greedy actions without exploration.

## Method and findings

The basic DQN agents use a separate target network and replay buffer. Task 3 selects the next action with the online network and evaluates it with the target network (Double DQN), samples replay transitions by TD-error priority, and accumulates rewards over three steps. The reported CartPole evaluation reached a reward of 500 in some episodes. For Pong, the enhanced agent reached high positive rewards earlier than the baseline in the reported runs, but evaluation rewards varied considerably. Because all three enhancements were combined in Task 3, these experiments do not isolate the effect of each enhancement.

See `LAB5_E94111253.pdf` for the training curves, evaluation screenshots, and discussion, if the report is included in the repository.
