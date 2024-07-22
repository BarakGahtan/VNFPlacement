import random

import pulp
import numpy as np
from matplotlib import pyplot as plt

from utils1.utils_functions import calculate_radius


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
    z = pulp.LpVariable.dicts("z", ((c, s) for c in range(num_clients) for s in range(num_functions)), 0, 1, pulp.LpContinuous)
    y = pulp.LpVariable.dicts("y", ((c, s) for c in range(num_clients) for s in range(num_servers)), 0, 2, pulp.LpContinuous)

    prob += pulp.lpSum([y[c, s] for c in range(num_clients) for s in range(num_servers)])

    for s in range(num_servers):
        prob += pulp.lpSum([x[s, f] for f in range(num_functions)]) <= 2, f"Function_Placement_Constraint_Server_{s}"

    for s in range(num_servers):
        prob += pulp.lpSum([y[c, s] for c in range(num_clients)]) <= weights[s], f"Capacity_Constraint_Server_{s}"

    for c in range(num_clients):
        for s in range(num_servers):
            prob += y[c, s] <= pulp.lpSum([x[s, f] for f in client_demands[c] if np.linalg.norm(np.array(client_positions[c]) - np.array(server_positions[s])) <= radius]), f"Client_Demand_Constraint_{c}_{s}"

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
    y = pulp.LpVariable.dicts("y", ((c, s) for c in range(num_clients) for s in range(num_servers)), 0, 2, pulp.LpContinuous)

    prob += pulp.lpSum([y[c, s] for c in range(num_clients) for s in range(num_servers)])

    for s in range(num_servers):
        prob += pulp.lpSum([x[s, f] for f in range(num_functions)]) <= 2, f"Function_Placement_Constraint_Server_{s}"

    for s in range(num_servers):
        prob += pulp.lpSum([y[c, s] for c in range(num_clients)]) <= weights[s], f"Capacity_Constraint_Server_{s}"

    for c in range(num_clients):
        for s in range(num_servers):
            prob += y[c, s] <= pulp.lpSum([x[s, f] for f in client_demands[c] if np.linalg.norm(np.array(client_positions[c]) - np.array(server_positions[s])) <= radius]), f"Client_Demand_Constraint_{c}_{s}"

    prob.solve()

    results = {(s, f): round(pulp.value(x[s, f])) for s in range(num_servers) for f in range(num_functions)}
    client_served_by_server = {(c, s): round(pulp.value(y[c, s])) for c in range(num_clients) for s in range(num_servers)}

    # Extract results and round probabilistically
    results = {(s, f): pulp.value(x[s, f]) for s in range(num_servers) for f in range(num_functions)}
    client_served_by_server = {(c, s): pulp.value(y[c, s]) for c in range(num_clients) for s in range(num_servers)}
    server_placements_arr = np.array([[pulp.value(x[s, f]) for s in range(num_servers)] for f in range(num_functions)])
    prob_array = np.random.rand(num_servers, num_functions)
    arr = (server_placements_arr > prob_array).astype(int)
    placements = {s: [] for s in range(num_servers)}
    for (s, f), value in results.items():
        if random.random() < value:  # Probabilistic rounding
            placements[s].append(f)

    rounded_client_served_by_server = {(c, s): 1 if random.random() < value else 0 for (c, s), value in client_served_by_server.items()}

    return placements, rounded_client_served_by_server



def call_LP_solvers(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions, client_demands):
# Call the linear programming functions
    placements_not_rounded = fractional_linear_programming(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions, client_demands)
    placements_rounded, client_served_by_server = rounded_linear_programming(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions,
                                                                         client_demands)

    print("Not Rounded Placements:", placements_not_rounded)
    print("Rounded Placements:", placements_rounded)
    print("Client Served by Server:", client_served_by_server)