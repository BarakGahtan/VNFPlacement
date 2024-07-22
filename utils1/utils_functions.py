# The calculate_radius function computes the radius as a fraction of the maximum distance between any two servers.
# This ensures that each server can connect to some clients but not all.
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here the reward)
        reward = self.locals['rewards'][0]
        self.logger.record('reward', reward)
        return True

def calculate_radius(server_positions, fraction=0.5):
    # Calculate the pairwise distances between all servers
    distances = []
    for i in range(len(server_positions)):
        for j in range(i + 1, len(server_positions)):
            distance = np.linalg.norm(np.array(server_positions[i]) - np.array(server_positions[j]))
            distances.append(distance)
    # Set the radius as a fraction of the maximum distance
    max_distance = max(distances)
    radius = fraction * max_distance
    return radius