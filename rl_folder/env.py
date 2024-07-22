import gymnasium as gym
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt
from utils1.utils_functions import calculate_radius


class FunctionPlacementEnv(gym.Env):
    def __init__(self, num_servers, num_functions, subset_functions, num_clients, params, reward_params):
        super(FunctionPlacementEnv, self).__init__()
        self.num_servers = num_servers
        self.num_functions = num_functions
        self.subset_functions = subset_functions
        self.num_clients = num_clients
        self.params = params
        self.reward_params = reward_params
        self.steps = 0
        self.episode_number = 0

        self.action_space = spaces.MultiDiscrete([num_functions] * (num_servers * subset_functions))
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_servers * self.num_functions
                   + self.num_servers
                   + self.num_clients * self.num_servers
                   + self.num_clients * self.num_functions
                   + self.num_servers * 2
                   + 1,),
            dtype=np.float32
        )

        self.client_demands = np.zeros((self.num_clients, self.num_functions), dtype=np.int32)
        for i in range(self.num_clients):
            functions = np.random.choice(self.num_functions, self.subset_functions, replace=False)
            self.client_demands[i, functions] = 1

        self.reset()

    def reset(self, **kwargs):
        self.server_functions = np.zeros((self.num_servers, self.num_functions), dtype=np.int32)
        self.server_load = np.zeros(self.num_servers, dtype=np.float32)
        self.client_served = np.zeros((self.num_clients, self.num_servers), dtype=np.int32)
        self.server_positions = np.random.rand(self.num_servers, 2)
        self.radius = calculate_radius(self.server_positions, fraction=self.params.radius / 100)
        self.weights = np.random.uniform(self.params.w1 / 100 * self.num_clients, self.params.w2 / 100 * self.num_clients, self.num_servers).astype(int)
        self.client_positions = np.random.rand(self.num_clients, 2)
        self.steps = 0
        self.episode_number += 1
        # self.visualize_map()
        state = self._get_state()
        return state.astype(np.float32), {}

    def _get_state(self):
        state = np.concatenate([
            self.server_functions.flatten(),
            self.server_load,
            self.client_served.flatten(),
            self.client_demands.flatten(),
            self.server_positions.flatten(),
            np.array([self.radius / 10])
        ])
        return state.astype(np.float32)

    def calculate_reward(self):
        reward = 0
        for client in range(self.num_clients):
            client_served_functions = []
            required_functions = self.client_demands[client].nonzero()[0]
            for server in np.random.permutation(self.num_servers):
                if len(client_served_functions) == self.subset_functions: break
                if self._is_within_radius(client, server):
                    for f in required_functions:
                        if f in client_served_functions:
                            continue
                        if self.server_functions[server, f] == 1:
                            client_served_functions.append(f)
                            self.client_served[client, server] += 1
                            self.server_load[server] += 1 / self.weights[server]
                            reward += 1 / self.weights[server] # If results won't be good, maybe add the dict to the observation space.

                    # available_functions = self.server_functions[server, required_functions].sum()
                    # total_required = len(required_functions)
                    # partial_score = available_functions / total_required
                    # if partial_score > 0:
                    #     self.client_served[client, server] = #function
                    #     self.server_load[server] += partial_score / self.weights[server]
                    #     reward += partial_score / self.weights[server]
        overloaded_servers = (self.server_load > self.weights)
        reward -= np.sum(overloaded_servers) * self.reward_params['overload_penalty']
        load_variance = np.var(self.server_load)
        reward -= load_variance * self.reward_params['variance_penalty']
        return reward

    def step(self, action):
        # print(f"Action received: {action}")
        try:
            action = np.array(action).reshape((self.num_servers, self.subset_functions))
        except ValueError as e:
            print(f"Action reshape error: {e}")
            print(f"Action shape: {action.shape}, expected: ({self.num_servers}, {self.subset_functions})")
            raise e

        self.server_functions.fill(0)
        for server_idx in range(self.num_servers):
            for func_idx in action[server_idx]:
                self.server_functions[server_idx, func_idx] = 1

        self.server_load.fill(0)
        self.client_served.fill(0)
        reward = self.calculate_reward()

        self.client_positions = np.random.rand(self.num_clients, 2)
        self.steps += 1
        done = self.steps >= self.params.steps_count
        state = self._get_state()
        truncated = False

        return state.astype(np.float32), reward, done, truncated, {}

    def _is_within_radius(self, client, server):
        client_pos = self.client_positions[client]
        server_pos = self.server_positions[server]
        distance = np.linalg.norm(client_pos - server_pos)
        return distance <= self.radius

    def _is_client_served(self, client, clients_served_functions):
        if client not in clients_served_functions:
            return False
        return len(clients_served_functions[client]) ==  self.subset_functions

    def visualize_map(self):
        plt.figure(figsize=(10, 8))
        max_weight = max(self.weights)
        min_weight = min(self.weights)

        def normalize_size(weight):
            return 100 + (900 * (weight - min_weight) / (max_weight - min_weight))

        for i, (x, y) in enumerate(self.server_positions):
            normalized_size = normalize_size(self.weights[i])
            plt.scatter(x * 10, y * 10, color='blue', s=normalized_size, label='' if i == 0 else "")
            circle = plt.Circle((x * 10, y * 10), self.radius * 10, color='blue', fill=False, linestyle='dashed', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.scatter(self.client_positions[:, 0] * 10, self.client_positions[:, 1] * 10, color='red', label='Clients')
        plt.scatter([], [], color='blue', s=100, label='Server')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Server Locations with Coverage Radius and Server Weights - Episode {self.episode_number}')
        plt.legend(loc='upper right')
        plt.grid(True)
        # plt.savefig(f"/home/barak/PycharmProjects/dynamicVNFDRL/visual/episode_{self.episode_number}.png")
        plt.close()
