import random
import cvxpy as cp

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


def fractional_linear_programming(num_servers, num_functions, num_clients, weights, radius, client_positions, server_positions, client_demands):
    # Define the variables
    x = cp.Variable((num_servers, num_functions), boolean=True)
    z = cp.Variable((num_clients, num_functions), boolean=True)
    y = cp.Variable((num_clients, num_servers, num_functions), boolean=True)

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
        constraints.append(cp.sum(y[:, s, :]) <= weights[s])

    # Constraint: Client-Server-Function Constraint (Based on Distance)
    for c in range(num_clients):
        for s in range(num_servers):
            distance = np.linalg.norm(np.array(client_positions[c]) - np.array(server_positions[s]))
            for f in client_demands[c]:
                max_value = x[s, f] if distance <= radius else 0
                constraints.append(y[c, s, f] <= max_value)

    # Constraint: Client-Function Satisfaction Constraint
    for c in range(num_clients):
        for f in client_demands[c]:
            constraints.append(z[c, f] <= cp.sum(y[c, :, f]))

    # Define the problem
    prob = cp.Problem(objective, constraints)