# The calculate_radius function computes the radius as a fraction of the maximum distance between any two servers.
# This ensures that each server can connect to some clients but not all.
import numpy as np
from matplotlib import pyplot as plt
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


def create_clients_demand(num_clients, num_functions, subset_functions):
    client_demands = np.zeros((num_clients, num_functions), dtype=np.int32)
    for i in range(num_clients):
        functions = np.random.choice(num_functions, subset_functions, replace=False)
        client_demands[i, functions] = 1
    return client_demands

def visualize_map(weights,server_positions,client_positions,radius):
    plt.figure(figsize=(10, 8))
    max_weight = max(weights)
    min_weight = min(weights)

    def normalize_size(weight):
        return 100 + (900 * (weight - min_weight) / (max_weight - min_weight))

    for i, (x, y) in enumerate(server_positions):
        normalized_size = normalize_size(weights[i])
        plt.scatter(x * 10, y * 10, color='blue', s=normalized_size, label='' if i == 0 else "")
        circle = plt.Circle((x * 10, y * 10), radius * 10, color='blue', fill=False, linestyle='dashed', alpha=0.5)
        plt.gca().add_patch(circle)
    plt.scatter(client_positions[:, 0] * 10, client_positions[:, 1] * 10, color='red', label='Clients')
    plt.scatter([], [], color='blue', s=100, label='Server')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Server Locations with Coverage Radius and Server Weights')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    # plt.savefig(f"/home/barak/PycharmProjects/dynamicVNFDRL/visual/episode_{self.episode_number}.png")
    plt.close()