class Environment:

    name = "Environment"

    def __init__(self):
        pass


    def reset(self):
        """
        Reset the environment and return the initial state.
        """
        raise NotImplementedError


    def step(self, action):
        """
        Apply an action.

        Returns:
            next_state,
            reward,
            done
        """
        raise NotImplementedError


    def render(self):
        """
        Optional visualization.
        """
        pass


    def close(self):
        """
        Cleanup resources.
        """
        pass