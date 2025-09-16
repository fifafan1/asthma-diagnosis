# asthma-diagnosis
asthma-diagnosis Data Science project

## Reinforcement Learning Snake

The project now includes a reinforcement learning variant of the classic
Snake game.  The environment and agent are implemented from scratch and can
be found under `src/rl_snake`.  A Deep Q-Network (DQN) agent learns to play
Snake by interacting with the environment and improving through experience.

### Quick start

Train an agent for a small number of episodes from the command line:

```bash
python -m src.rl_snake.train --episodes 50 --width 10 --height 10 --checkpoint results/snake_agent.pt
```

Per-episode statistics can be saved with `--stats-path results/snake_stats.json`.
See `python -m src.rl_snake.train --help` for the full list of parameters.
