"""
Cliff Walking Environment

A classic grid world problem where an agent must navigate from start to goal
while avoiding a cliff. The grid is 4x12 with:
- Start position: (3, 0)
- Goal position: (3, 11)
- Cliff: (3, 1) through (3, 10) - falling off gives -100 reward
- Each step costs -1

Actions: 0=Up, 1=Right, 2=Down, 3=Left
States: (row, col) encoded as row * 12 + col
"""

from archid.reinforcement_learning.environment import Environment


class CliffWalking(Environment):
    name = "CliffWalking"

    def __init__(self):
        super().__init__()

        self.nrows = 4
        self.ncols = 12

        self.start_pos = (3, 0)
        self.goal_pos = (3, 11)

        self.cliff_positions = [
            (3, i)
            for i in range(1, 11)
        ]

        self.distance_reward = 0.10

        self.agent_pos = None

        self.reset()

    def _pos_to_state(self, row, col):
        return row * self.ncols + col

    def _state_to_pos(self, state):
        return divmod(state, self.ncols)

    def _distance_to_goal(self, pos):
        row, col = pos
        goal_row, goal_col = self.goal_pos

        return (
            abs(row - goal_row)
            + abs(col - goal_col)
        )

    def reset(self):
        self.agent_pos = self.start_pos

        return self._pos_to_state(*self.agent_pos)

    def step(self, action):

        # ----------------------------------------
        # Distance before action
        # ----------------------------------------

        old_distance = self._distance_to_goal(
            self.agent_pos
        )

        row, col = self.agent_pos

        # ----------------------------------------
        # Apply action
        # ----------------------------------------

        if action == 0:  # Up
            row = max(0, row - 1)

        elif action == 1:  # Right
            col = min(self.ncols - 1, col + 1)

        elif action == 2:  # Down
            row = min(self.nrows - 1, row + 1)

        elif action == 3:  # Left
            col = max(0, col - 1)

        self.agent_pos = (row, col)

        next_state = self._pos_to_state(
            row,
            col
        )

        # ----------------------------------------
        # Cliff
        # ----------------------------------------

        if self.agent_pos in self.cliff_positions:

            reward = -100
            done = True

            self.agent_pos = self.start_pos

            next_state = self._pos_to_state(
                *self.start_pos
            )

        # ----------------------------------------
        # Goal
        # ----------------------------------------

        elif self.agent_pos == self.goal_pos:

            reward = 0
            done = True

        # ----------------------------------------
        # Normal movement
        # ----------------------------------------

        else:

            new_distance = self._distance_to_goal(
                self.agent_pos
            )

            distance_change = (
                old_distance - new_distance
            )

            reward = (
                -1
                + self.distance_reward
                * distance_change
            )

            done = False

        return next_state, reward, done

    def render(self):

        grid = [
            ['.' for _ in range(self.ncols)]
            for _ in range(self.nrows)
        ]

        for row, col in self.cliff_positions:
            grid[row][col] = 'C'

        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'

        if self.agent_pos not in self.cliff_positions:
            grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        for row in grid:
            print(' '.join(row))

        print()