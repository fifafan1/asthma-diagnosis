"""Command line interface for training the Snake reinforcement learning agent."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch

from .agent import EpisodeStats, train_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--width", type=int, default=20, help="Board width in cells")
    parser.add_argument("--height", type=int, default=20, help="Board height in cells")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=512, help="Size of experience replay batches")
    parser.add_argument("--memory-size", type=int, default=50_000, help="Replay buffer capacity")
    parser.add_argument("--hidden-size", type=int, default=128, help="Number of neurons in the hidden layers")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Minimum epsilon for exploration")
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.995, help="Multiplicative epsilon decay applied each episode"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per episode")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="If provided, save the trained model weights to this path",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        help="Optional JSON file where per-episode statistics will be stored",
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default=None, help="Device to run the neural network on"
    )
    return parser.parse_args()


def save_stats(stats: List[EpisodeStats], path: Path) -> None:
    """Persist the training statistics to a JSON file."""

    serialisable = [asdict(stat) for stat in stats]
    path.write_text(json.dumps(serialisable, indent=2))


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or parse_args()

    env_kwargs = {"width": parsed.width, "height": parsed.height}
    agent_kwargs = {
        "learning_rate": parsed.learning_rate,
        "gamma": parsed.gamma,
        "batch_size": parsed.batch_size,
        "memory_size": parsed.memory_size,
        "hidden_size": parsed.hidden_size,
        "epsilon_start": parsed.epsilon_start,
        "epsilon_end": parsed.epsilon_end,
        "epsilon_decay": parsed.epsilon_decay,
    }
    if parsed.device is not None:
        agent_kwargs["device"] = torch.device(parsed.device)

    stats, agent = train_agent(
        num_episodes=parsed.episodes,
        max_steps_per_episode=parsed.max_steps,
        seed=parsed.seed,
        env_kwargs=env_kwargs,
        agent_kwargs=agent_kwargs,
    )

    print("Training complete. Final episode stats:")
    if stats:
        last = stats[-1]
        print(
            f"Episode {last.episode}: score={last.score}, total_reward={last.total_reward:.1f}, "
            f"epsilon={last.epsilon:.3f}, avg_loss={last.average_loss:.4f}"
        )

    if parsed.checkpoint:
        checkpoint_path = Path(parsed.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint_path))
        print(f"Model checkpoint saved to {checkpoint_path}")

    if parsed.stats_path:
        stats_path = Path(parsed.stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        save_stats(stats, stats_path)
        print(f"Episode statistics stored at {stats_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
