import numpy as np
from sklearn.cluster import KMeans

from LP.LP_algo import cvxpy_fun
from utils1.utils_functions import calculate_radius


def optimize_clusters(params, num_clusters,client,  max_iterations=100):
    # Initialize clusters with KMeans
    server_positions = np.random.rand(params['num_servers'], 2)
    client_positions = np.random.rand(params['num_clients'], 2)

    kmeans = KMeans(n_clusters=num_clusters)
    server_clusters = kmeans.fit_predict(server_positions)
    client_clusters = np.argmin(np.linalg.norm(
        client_positions[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :], axis=2), axis=1)

    best_placements = {}
    best_prob_value = -np.inf

    for iteration in range(max_iterations):
        current_placements = {}
        current_prob_value = 0

        for cluster_id in range(num_clusters):
            # Filter servers and clients for the current cluster
            cluster_servers = np.where(server_clusters == cluster_id)[0]
            cluster_clients = np.where(client_clusters == cluster_id)[0]

            if len(cluster_servers) == 0 or len(cluster_clients) == 0:
                continue  # Skip empty clusters

            num_servers = len(cluster_servers)
            num_clients = len(cluster_clients)

            cluster_server_positions = server_positions[cluster_servers]
            cluster_client_positions = client_positions[cluster_clients]

            cluster_weights = params['weights'][cluster_servers]
            cluster_client_demands = params['client_demands'][cluster_clients, :]

            radius = calculate_radius(cluster_server_positions, fraction=params['params'].radius / 100)

            # Solve LP problem for the current cluster
            placements, prob_value = cvxpy_fun(num_servers, params['num_functions'], num_clients,
                                               cluster_weights, radius, cluster_client_positions,
                                               cluster_server_positions, cluster_client_demands)

            current_placements[cluster_id] = placements
            current_prob_value += prob_value

        # Compare and update the best solution found
        if current_prob_value > best_prob_value:
            best_prob_value = current_prob_value
            best_placements = current_placements

        # Adjust clustering for next iteration (this is where you can get creative)
        # One simple approach could be to move clients between clusters based on current distances.
        client_clusters = adjust_clusters_based_on_prob_value(client_positions, kmeans.cluster_centers_, server_positions, best_placements)

    return best_placements, best_prob_value

def adjust_clusters_based_on_prob_value(client_positions, cluster_centers, server_positions, placements):
    # Example of adjusting clusters based on the objective values:
    # Move clients to the cluster of the nearest server with the highest placement value.

    new_client_clusters = np.argmin(np.linalg.norm(
        client_positions[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2), axis=1)

    return new_client_clusters

# Example usage
params = {
    'num_servers': 25,
    'num_clients': 100,
    'num_functions': 5,
    'subset_functions': 3,
    'params': {
        'radius': 10,
        'w1': 5,
        'w2': 15
    }
}

client_demands = create_clients_demand(params['num_clients'], params['num_functions'], params['subset_functions'])
server_functions = np.zeros((params['num_servers'], params['num_functions']), dtype=np.int32)
server_load = np.zeros(params['num_servers'], dtype=np.float32)
client_served = np.zeros((params['num_clients'], params['num_servers']), dtype=np.int32)
server_positions = np.random.rand(params['num_servers'], 2)
radius = calculate_radius(server_positions, fraction=params['params'].radius / 100)
weights = np.random.uniform(params['params'].w1 / 100 * params['num_clients'], params['params'].w2 / 100 * params['num_clients'], params['num_servers']).astype(int)
client_positions = np.random.rand(params['num_clients'], 2)
num_clusters = 5  # Configurable number of clusters
best_placements, best_prob_value = optimize_clusters(params, num_clusters, client_positions, server_positions, weights, client_demands)

# The results are stored in best_placements and best_prob_value
