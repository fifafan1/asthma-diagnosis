from src.rl_snake.environment import SnakeGameEnv
from src.rl_snake.agent import train_agent


def test_environment_basic_step():
    env = SnakeGameEnv(width=5, height=5, seed=42, max_steps_without_food=25)
    state = env.reset()
    assert state.shape == (11,)

    next_state, reward, done, info = env.step(0)
    assert next_state.shape == (11,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "score" in info


def test_train_agent_runs_quickly():
    stats, agent = train_agent(
        num_episodes=3,
        max_steps_per_episode=60,
        seed=123,
        env_kwargs={"width": 6, "height": 6, "max_steps_without_food": 40},
        agent_kwargs={
            "hidden_size": 32,
            "batch_size": 32,
            "memory_size": 1_000,
            "epsilon_start": 0.6,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.8,
            "learning_rate": 5e-4,
        },
    )

    assert len(stats) == 3
    assert stats[-1].episode == 3
    assert agent.model is not None
