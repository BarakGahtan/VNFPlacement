import numpy as np
import torch

from partition_model import cluster_and_solve_dynamic
from utils1 import input_parser
from utils1.utils_functions import calculate_radius, visualize_map, create_clients_demand

if __name__ == "__main__":
    parsed_args = input_parser.Parser()
    opts = parsed_args.parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {
        'num_servers': opts.servers_cnt,
        'num_functions': opts.possible_func,
        'subset_functions': opts.demanded_func,
        'num_clients': opts.clients_cnt,
        'params': opts
    }
    client_demands = create_clients_demand(params['num_clients'], params['num_functions'], params['subset_functions'])
    server_functions = np.zeros((params['num_servers'], params['num_functions']), dtype=np.int32)
    server_load = np.zeros(params['num_servers'], dtype=np.float32)
    client_served = np.zeros((params['num_clients'], params['num_servers']), dtype=np.int32)
    server_positions = np.random.rand(params['num_servers'], 2)
    radius = calculate_radius(server_positions, fraction=params['params'].radius / 100)
    weights = np.random.uniform(params['params'].w1 / 100 * params['num_clients'], params['params'].w2 / 100 * params['num_clients'], params['num_servers']).astype(int)
    client_positions = np.random.rand(params['num_clients'], 2)
    visualize_map(weights,server_positions,client_positions,radius)
    cluster_and_solve_dynamic(params, server_positions, radius, weights, client_demands)
    print("done")