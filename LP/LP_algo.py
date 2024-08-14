import random
import cvxpy as cp

import pulp
import numpy as np
import torch
from matplotlib import pyplot as plt

from utils1 import input_parser
from utils1.utils_functions import calculate_radius, create_clients_demand


# x_sf is a binary variable indicating whether function f is placed on server s.
# Objective: Maximize the sum of clients served by servers that have the required functions placed and are within the radius.
# Constraints: Each server can only host up to its weight (capacity) in terms of functions placed.
# The problem is solved as a continuous LP problem. The resulting fractional assignments are rounded to binary values (0 or 1)
# This implementation assumes that each client c demands a set of functions.
# The radius constraint is applied to ensure only servers within the specified distance can serve the client.
# The results are rounded to obtain a feasible integer solution, which may not be optimal but should be near-optimal.


# Constraints:
# Each server can place only one function.
# Server capacity limits the number of clients served.
# Clients must be within the radius of a server and the server must host the required function.


# LP function without rounding
def fractional_linear_programming(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions, client_demands):
    prob = pulp.LpProblem("Function_Placement", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_servers) for j in range(num_functions)), 0, 1, pulp.LpContinuous)
    z = pulp.LpVariable.dicts("z", ((c, f) for c in range(num_clients) for f in range(num_functions)), 0, 1, pulp.LpContinuous)
    y = pulp.LpVariable.dicts("y", ((c, s, f) for c in range(num_clients) for s in range(num_servers) for f in range(num_functions)), 0, 1, pulp.LpContinuous)

    prob += pulp.lpSum([z[c, f] for c in range(num_clients) for f in range(num_functions)])

    for s in range(num_servers):
        prob += pulp.lpSum([x[s, f] for f in range(num_functions)]) <= 2, f"Function_Placement_Constraint_Server_{s}"


    for c in range(num_clients):
        for f in range(num_functions):
            max_value = 1 if f in client_demands[c] else 0
            prob += z[c, f] <= max_value, f"Client_Function_Constraint_{c}_{f}"

    for s in range(num_servers):
        prob += pulp.lpSum([y[c, s, f] for c in range(num_clients) for f in range(num_functions)]) <= weights[s], f"Capacity_Constraint_Server_{s}"

    for c in range(num_clients):
        for s in range(num_servers):
            for f in client_demands[c]:
                max_value = x[s, f] if np.linalg.norm(np.array(client_positions[c]) - np.array(server_positions[s])) <= radius else 0
                prob += y[c, s, f] <= max_value, f"Client_Function_Constraint_{c}_{s}_{f}"

    for c in range(num_clients):
        for f in client_demands[f]:
            prob += z[c,f] < pulp.lpSum([y[c, s, f] for s in range(num_servers)]), f"Client_Function_Constraint_{c}_{f}"

    prob.solve()

    results = {(s, f): pulp.value(x[s, f]) for s in range(num_servers) for f in range(num_functions)}
    placements = {s: [] for s in range(num_servers)}
    for (s, f), value in results.items():
        if value > 0:
            placements[s].append((f, value))

    return placements, pulp.value(prob.objective)

# LP function with rounding
def rounded_linear_programming(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions, client_demands):
    prob = pulp.LpProblem("Function_Placement", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_servers) for j in range(num_functions)), 0, 1, pulp.LpContinuous)
    z = pulp.LpVariable.dicts("z", ((c, f) for c in range(num_clients) for f in range(num_functions)), 0, 1, pulp.LpContinuous)
    y = pulp.LpVariable.dicts("y", ((c, s, f) for c in range(num_clients) for s in range(num_servers) for f in range(num_functions)), 0, 1, pulp.LpContinuous)

    prob += pulp.lpSum([z[c, f] for c in range(num_clients) for f in range(num_functions)])

    for s in range(num_servers):
        prob += pulp.lpSum([x[s, f] for f in range(num_functions)]) <= 2, f"Function_Placement_Constraint_Server_{s}"

    for c in range(num_clients):
        for f in range(num_functions):
            max_value = 1 if f in client_demands[c] else 0
            prob += z[c, f] <= max_value, f"Client_Function_Constraint_{c}_{f}"

    for s in range(num_servers):
        prob += pulp.lpSum([y[c, s, f] for c in range(num_clients) for f in range(num_functions)]) <= weights[s], f"Capacity_Constraint_Server_{s}"

    for c in range(num_clients):
        for s in range(num_servers):
            for f in client_demands[c]:
                max_value = x[s, f] if np.linalg.norm(np.array(client_positions[c]) - np.array(server_positions[s])) <= radius else 0
                prob += y[c, s, f] <= max_value, f"Client_Function_Constraint_{c}_{s}_{f}"

    for c in range(num_clients):
        for f in client_demands[f]:
            prob += z[c, f] < pulp.lpSum([y[c, s, f] for s in range(num_servers)]), f"Client_Function_Constraint_{c}_{f}"

    prob.solve()

    # Extract results and round probabilistically
    client_served_by_server = {(c, s, f): pulp.value(y[c, s]) for c in range(num_clients) for s in range(num_servers)}
    server_placements_arr = np.array([[pulp.value(x[s, f]) for s in range(num_servers)] for f in range(num_functions)])
    while True:
        prob_array = np.random.rand(num_servers, num_functions)
        placements = (server_placements_arr >= prob_array).astype(int)
        if np.all(np.sum(placements, axis=0) <= 2):
            break

    rounded_client_served_by_server = {(c, s, f): 1 if random.random() < value and placements[s, f] == 1 else 0 for (c, s, f), value in client_served_by_server.items()}

    return placements, rounded_client_served_by_server, sum(rounded_client_served_by_server.values())



def call_LP_solvers(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions, client_demands):
# Call the linear programming functions
    placements_not_rounded = fractional_linear_programming(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions, client_demands)
    placements_rounded, client_served_by_server = rounded_linear_programming(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions,
                                                                         client_demands)

    print("Not Rounded Placements:", placements_not_rounded)
    print("Rounded Placements:", placements_rounded)
    print("Client Served by Server:", client_served_by_server)


def cvxpy_fun(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions, client_demands):
    # Define the variables
    # Define the variables
    x = cp.Variable((num_servers, num_functions))
    z = cp.Variable((num_clients, num_functions))

    # y is reshaped as a 2D variable with the dimensions (num_clients, num_servers * num_functions)
    y = cp.Variable((num_clients, num_servers * num_functions))

    # Define the objective function
    objective = cp.Maximize(cp.sum(z))

    # Define the constraints
    constraints = []

    # Constraint: Server Function Placement Constraint
    for s in range(num_servers):
        constraints.append(cp.sum(x[s, :]) <= 2)

    # Constraint: Client Function Demand Constraint
    for c in range(num_clients):
        for f in range(num_functions):
            max_value = 1 if f in client_demands[c] else 0
            constraints.append(z[c, f] <= max_value)

    # Constraint: Server Capacity Constraint
    for s in range(num_servers):
        constraints.append(cp.sum(y[:, s * num_functions:(s + 1) * num_functions]) <= weights[s])

    # Constraint: Client-Server-Function Constraint (Based on Distance)
    for c in range(num_clients):
        for s in range(num_servers):
            distance = np.linalg.norm(np.array(client_positions[c]) - np.array(server_positions[s]))
            for f in client_demands[c]:
                idx = s * num_functions + f
                max_value = x[s, f] if distance <= radius else 0
                constraints.append(y[c, idx] <= max_value)

    # Constraint: Client-Function Satisfaction Constraint
    for c in range(num_clients):
        for f in client_demands[c]:
            indices = [s * num_functions + f for s in range(num_servers)]
            constraints.append(z[c, f] <= cp.sum(y[c, indices]))

    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    # Extract the results
    placements = {s: [] for s in range(num_servers)}
    for s in range(num_servers):
        for f in range(num_functions):
            if x[s, f].value > 0:
                placements[s].append((f, x[s, f].value))

    return placements, prob.value


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
placements, value = cvxpy_fun(params['num_servers'], params['num_functions'], params['num_clients'], weights, radius, client_positions, server_positions, client_demands)
place_1, value_1 = fractional_linear_programming(params['num_servers'], params['num_functions'], params['num_clients'], weights, radius, client_positions, server_positions, client_demands)
x  =5

# TODO: run and compare both LP
# TODO: write the LP in theusis and explain the constraints
# TODO: insert it into the training loop.
# TODO: talk to Danny about maybe additional features.
