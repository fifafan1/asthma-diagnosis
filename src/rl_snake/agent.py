"""Deep Q-learning agent capable of learning to play Snake."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .environment import SnakeGameEnv
from .model import LinearQNet, ReplayBuffer, Transition, build_optimizer


@dataclass
class EpisodeStats:
    """Aggregate training statistics for a single episode."""

    episode: int
    score: int
    total_reward: float
    epsilon: float
    average_loss: float


class DQNAgent:
    """Deep Q-learning agent with experience replay."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        gamma: float = 0.9,
        learning_rate: float = 1e-3,
        memory_size: int = 50_000,
        batch_size: int = 512,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device: Optional[torch.device] = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LinearQNet(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = build_optimizer(self.model, learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(memory_size)

    # ------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        """Return an action index following an epsilon-greedy policy."""

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        transition = Transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.push(transition)

    def learn(self) -> Optional[float]:
        """Sample a batch from memory and update the Q-network."""

        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.device)
        dones_tensor = torch.from_numpy(dones.astype(np.float32)).float().to(self.device)

        # Current Q estimates for the taken actions
        q_values = self.model(states_tensor)
        state_action_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model(next_states_tensor).max(1)[0]
            target_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        loss = self.criterion(state_action_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the model weights to disk."""

        torch.save(self.model.state_dict(), path)


def train_agent(
    num_episodes: int = 200,
    max_steps_per_episode: Optional[int] = None,
    seed: Optional[int] = None,
    callback: Optional[Callable[[EpisodeStats], None]] = None,
    env_kwargs: Optional[Dict] = None,
    agent_kwargs: Optional[Dict] = None,
) -> Tuple[List[EpisodeStats], DQNAgent]:
    """Train the DQN agent on the Snake environment.

    Parameters
    ----------
    num_episodes:
        Number of episodes to train for.
    max_steps_per_episode:
        Optional limit for the number of steps in a single episode.  If not
        provided, a sensible default based on the board size is used.
    seed:
        Seed used for the environment's random number generator.
    callback:
        Optional callable invoked after each episode with the latest
        :class:`EpisodeStats` instance.
    env_kwargs, agent_kwargs:
        Additional keyword arguments forwarded to :class:`SnakeGameEnv` and
        :class:`DQNAgent` respectively.
    """

    env_params = env_kwargs or {}
    env = SnakeGameEnv(seed=seed, **env_params)
    state = env.reset()
    state_size = state.shape[0]
    action_size = 3

    agent_params = agent_kwargs or {}
    agent = DQNAgent(state_size=state_size, action_size=action_size, **agent_params)

    max_steps = max_steps_per_episode or env.width * env.height * 4

    stats: List[EpisodeStats] = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        running_losses: List[float] = []
        info: Dict[str, float] = {"score": 0}

        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                running_losses.append(loss)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.update_epsilon()

        average_loss = float(np.mean(running_losses)) if running_losses else 0.0
        episode_stats = EpisodeStats(
            episode=episode,
            score=int(info.get("score", 0)),
            total_reward=float(total_reward),
            epsilon=float(agent.epsilon),
            average_loss=average_loss,
        )
        stats.append(episode_stats)

        if callback is not None:
            callback(episode_stats)

    return stats, agent


__all__ = ["DQNAgent", "EpisodeStats", "train_agent"]
