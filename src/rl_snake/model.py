"""Neural network utilities for the Snake reinforcement learning agent."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LinearQNet(nn.Module):
    """A small fully connected network used to approximate the Q-function."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size buffer implementing experience replay."""

    def __init__(self, capacity: int) -> None:
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if batch_size > len(self.memory):
            raise ValueError("Not enough elements in replay buffer to sample the batch")
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.memory[idx] for idx in indices))
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.memory)


def build_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    """Construct the Adam optimiser used by the agent."""

    return optim.Adam(model.parameters(), lr=learning_rate)


__all__ = ["LinearQNet", "ReplayBuffer", "Transition", "build_optimizer"]
