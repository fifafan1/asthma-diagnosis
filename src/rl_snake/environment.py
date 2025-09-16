"""Snake game environment for reinforcement learning.

This module implements a minimalistic version of the classic Snake game
that can be used as an OpenAI Gym-like environment.  The implementation is
self contained so it can run without any third party RL libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from collections import deque
from typing import Deque, Iterable, Optional, Tuple

import numpy as np


class Direction(Enum):
    """Cardinal directions that the snake can move in."""

    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4


@dataclass(frozen=True)
class Point:
    """Immutable representation of a location on the grid."""

    x: int
    y: int


class SnakeGameEnv:
    """A simple Snake environment tailored for reinforcement learning.

    The environment exposes an API similar to the OpenAI Gym interface with
    ``reset`` and ``step`` methods.  Observations are compact feature vectors
    describing the immediate surroundings of the snake and the relative
    position of the food.  Actions are encoded as integers representing how
    the snake should turn relative to its current direction:

    ``0`` – continue straight, ``1`` – turn right, ``2`` – turn left.
    """

    metadata = {"render.modes": ["ascii"]}

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        seed: Optional[int] = None,
        max_steps_without_food: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.block_size = 1
        self.random = random.Random(seed)
        self.max_steps_without_food = (
            max_steps_without_food if max_steps_without_food is not None else width * height
        )

        # Internal state placeholders initialised in ``reset``
        self.snake: Deque[Point]
        self.head: Point
        self.food: Point
        self.direction: Direction
        self.score: int
        self.steps_since_last_food: int

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""

        centre_x = self.width // 2
        centre_y = self.height // 2
        self.direction = Direction.RIGHT
        self.head = Point(centre_x, centre_y)
        self.snake = deque(
            [
                self.head,
                Point(centre_x - 1, centre_y),
                Point(centre_x - 2, centre_y),
            ]
        )
        self.score = 0
        self.steps_since_last_food = 0
        self._place_food()
        return self._get_state()

    def step(self, action: int | Iterable[int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time step within the environment.

        Parameters
        ----------
        action:
            Either an integer ``0``..``2`` or an iterable representing an
            one-hot encoded action.
        """

        action_idx = self._interpret_action(action)
        self._move(action_idx)
        self.snake.appendleft(self.head)
        self.steps_since_last_food += 1

        reward = 0.0
        done = False

        if self._is_collision(self.head) or self.steps_since_last_food > self.max_steps_without_food:
            done = True
            reward = -10.0
            return self._get_state(), reward, done, {"score": self.score}

        if self.head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_since_last_food = 0
            self._place_food()
        else:
            # Remove the tail segment to keep the snake length constant when
            # not eating food.
            self.snake.pop()
            # Small time penalty that encourages the agent to find food quickly.
            reward = -0.1

        return self._get_state(), reward, done, {"score": self.score}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _place_food(self) -> None:
        """Randomly place food on the grid avoiding the snake's body."""

        while True:
            x = self.random.randrange(0, self.width)
            y = self.random.randrange(0, self.height)
            food_candidate = Point(x, y)
            if food_candidate not in self.snake:
                self.food = food_candidate
                return

    def _interpret_action(self, action: int | Iterable[int]) -> int:
        if isinstance(action, Iterable) and not isinstance(action, (str, bytes)):
            action_list = list(action)
            if not action_list:
                raise ValueError("Action iterable must not be empty")
            action_idx = int(np.argmax(action_list))
        else:
            action_idx = int(action)
        if action_idx not in (0, 1, 2):
            raise ValueError("Action must be 0, 1 or 2")
        return action_idx

    def _move(self, action_idx: int) -> None:
        """Update the snake's head position based on the selected action."""

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action_idx == 0:  # straight
            new_direction = clock_wise[idx]
        elif action_idx == 1:  # turn right -> clockwise
            new_direction = clock_wise[(idx + 1) % 4]
        else:  # action_idx == 2 -> turn left / anti-clockwise
            new_direction = clock_wise[(idx - 1) % 4]

        self.direction = new_direction

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
        else:  # self.direction == Direction.DOWN
            y += self.block_size
        self.head = Point(x, y)

    # Observation helpers ------------------------------------------------
    def _get_state(self) -> np.ndarray:
        """Return the agent-centric state representation."""

        danger_straight, danger_right, danger_left = self._danger()

        dir_left = int(self.direction == Direction.LEFT)
        dir_right = int(self.direction == Direction.RIGHT)
        dir_up = int(self.direction == Direction.UP)
        dir_down = int(self.direction == Direction.DOWN)

        food_left = int(self.food.x < self.head.x)
        food_right = int(self.food.x > self.head.x)
        food_up = int(self.food.y < self.head.y)
        food_down = int(self.food.y > self.head.y)

        state = np.array(
            [
                danger_straight,
                danger_right,
                danger_left,
                dir_left,
                dir_right,
                dir_up,
                dir_down,
                food_left,
                food_right,
                food_up,
                food_down,
            ],
            dtype=np.float32,
        )
        return state

    def _danger(self) -> Tuple[int, int, int]:
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        dir_straight = clock_wise[idx]
        dir_right = clock_wise[(idx + 1) % 4]
        dir_left = clock_wise[(idx - 1) % 4]

        danger_straight = int(self._will_collide(dir_straight))
        danger_right = int(self._will_collide(dir_right))
        danger_left = int(self._will_collide(dir_left))
        return danger_straight, danger_right, danger_left

    def _will_collide(self, direction: Direction) -> bool:
        next_point = self._next_point(direction)
        return self._is_collision(next_point)

    def _next_point(self, direction: Direction) -> Point:
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.UP:
            y -= self.block_size
        else:  # direction == Direction.DOWN
            y += self.block_size
        return Point(x, y)

    def _is_collision(self, point: Point) -> bool:
        if point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height:
            return True
        # Skip the last element in the deque when checking collisions because
        # the tail moves forward unless the snake just ate food.
        return point in list(self.snake)[:-1]

    # Rendering -----------------------------------------------------------
    def render(self) -> str:
        """Return an ASCII representation of the grid.

        The method is primarily intended for debugging and unit tests.  It
        produces a human readable snapshot showing the snake and the food
        location.
        """

        grid = [[" "] * self.width for _ in range(self.height)]
        for segment in self.snake:
            grid[segment.y][segment.x] = "o"
        grid[self.head.y][self.head.x] = "@"
        grid[self.food.y][self.food.x] = "*"
        return "\n".join("".join(row) for row in grid)


__all__ = ["SnakeGameEnv", "Direction", "Point"]
