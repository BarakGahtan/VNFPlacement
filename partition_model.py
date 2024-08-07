import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.data import Data
from torch.optim import Adam, lr_scheduler
import numpy as np
from sklearn.manifold import TSNE
from LP.LP_algo import fractional_linear_programming
from utils1.utils_functions import create_clients_demand, calculate_radius


# Define a custom GNN layer for message passing
class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='mean')  # "mean" aggregation
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class GNNClustering(nn.Module):
    def __init__(self, in_channels, out_channels, num_servers, num_clients):
        super(GNNClustering, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, out_channels)
        self.num_servers = num_servers
        self.num_clients = num_clients

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

    def loss(self, output, client_positions, server_positions, radius, server_weights):
        client_clusters = output[:self.num_clients]
        server_clusters = output[self.num_clients:]

        # Cluster separation loss
        separation_loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                distance = torch.norm(client_clusters[i] - client_clusters[j])
                separation_loss += torch.exp(-distance)

        # Radius and server weight constraints
        radius_loss = torch.tensor(0.0, dtype=torch.float32)
        for client_idx in range(self.num_clients):
            for server_idx in range(self.num_servers):
                distance = torch.norm(client_positions[client_idx] - server_positions[server_idx])
                if distance > radius:
                    radius_loss += F.mse_loss(client_clusters[client_idx], server_clusters[server_idx]) * distance

        # Server weight balance constraint
        weight_loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(self.num_servers):
            for j in range(self.num_servers):
                if i != j:
                    weight_diff = server_weights[i] - server_weights[j]
                    weight_loss += F.mse_loss(server_clusters[i], server_clusters[j]) * weight_diff.abs()

        # Total loss
        total_loss = separation_loss + radius_loss + weight_loss
        return total_loss


def prepare_graph_data(server_positions, client_positions, server_weights, radius):
    num_servers = server_positions.shape[0]
    num_clients = client_positions.shape[0]

    # Combine server positions and weights into node features
    server_features = np.hstack((server_positions, server_weights.reshape(-1, 1)))
    client_features = client_positions

    # Normalize the features
    scaler = StandardScaler()
    server_features = scaler.fit_transform(server_features)
    client_features = scaler.fit_transform(client_features)

    # Create edge index based on radius constraint
    edge_index = []
    for i in range(num_clients):
        for j in range(num_servers):
            if np.linalg.norm(client_positions[i] - server_positions[j]) <= radius:
                edge_index.append([i, num_clients + j])
                edge_index.append([num_clients + j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create a list of nodes including servers and clients
    x = torch.tensor(np.vstack((client_features, server_features)), dtype=torch.float)

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    return data


def aggregate_solutions(solutions, num_servers, num_clients):
    """
    Aggregate the solutions from each cluster into a final global solution.

    Parameters:
    - solutions: List of dictionaries containing the solution for each cluster. Each dictionary contains:
      - 'server_functions': A binary matrix of shape (num_servers, num_functions) indicating function placement.
      - 'server_load': An array of shape (num_servers,) indicating the load on each server.
      - 'client_served': A binary matrix of shape (num_clients, num_servers) indicating client assignments.
    - num_servers: The total number of servers.
    - num_clients: The total number of clients.

    Returns:
    - final_solution: A dictionary containing the aggregated solution:
      - 'server_functions': A binary matrix of shape (num_servers, num_functions) indicating function placement.
      - 'server_load': An array of shape (num_servers,) indicating the load on each server.
      - 'client_served': A binary matrix of shape (num_clients, num_servers) indicating client assignments.
    """
    num_functions = solutions[0]['server_functions'].shape[1]

    # Initialize the final solution
    final_server_functions = np.zeros((num_servers, num_functions), dtype=np.int32)
    final_server_load = np.zeros(num_servers, dtype=np.float32)
    final_client_served = np.zeros((num_clients, num_servers), dtype=np.int32)

    # Aggregate solutions from each cluster
    for solution in solutions:
        final_server_functions += solution['server_functions']
        final_server_load += solution['server_load']
        final_client_served += solution['client_served']

    # Ensure no server exceeds its capacity
    final_server_functions = np.clip(final_server_functions, 0, 1)
    final_server_load = np.clip(final_server_load, 0, 1)
    final_client_served = np.clip(final_client_served, 0, 1)

    final_solution = {
        'server_functions': final_server_functions,
        'server_load': final_server_load,
        'client_served': final_client_served
    }
    return final_solution


# Client_labels serve as these target values, helping the GNN learn how to cluster the clients and servers effectively. These labels indicate which cluster each client belongs to,
# allowing the model to learn the patterns and features that define each cluster. client_labels are initialized randomly for training purposes using
# torch.randint(0, num_clusters, (params['num_clients'],)). This is a placeholder and assumes that during training, you have some way to assign initial cluster labels to the clients.


def cluster_and_solve_dynamic(params, server_positions, radius, server_weights, client_demands):
    num_clusters = int(params['num_servers'] / 5) # Example number of clusters

    model = GNNClustering(in_channels=server_positions.shape[1] + 1, out_channels=num_clusters, num_servers=params['num_servers'],
                          num_clients=params['num_clients'])
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    losses, embeddings = [], []
    model.train()
    for epoch in range(params['params'].epochs):  # Training epochs
        client_positions = np.random.rand(params['num_clients'], 2)  # Update client positions each epoch
        data = prepare_graph_data(server_positions, client_positions, server_weights, radius)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = model.loss(output, torch.tensor(client_positions, dtype=torch.float32),
                          torch.tensor(server_positions, dtype=torch.float32), radius, torch.tensor(server_weights, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        embeddings.append(output.detach().cpu().numpy())
        if epoch % 10 == 0:  # Print every 10 epochs
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), 'gnn_clustering_model.pth')

    # Visualize the final embeddings
    final_embeddings = embeddings[-1]
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(final_embeddings)
    plt.figure(figsize=(10, 5))
    plt.scatter(reduced_embeddings[:params['num_clients'], 0], reduced_embeddings[:params['num_clients'], 1], c=client_clusters, cmap='viridis', label='Clients')
    plt.scatter(reduced_embeddings[params['num_clients']:, 0], reduced_embeddings[params['num_clients']:, 1], c='red', label='Servers')
    plt.legend()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D Visualization of Node Embeddings')
    plt.show()


    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

    model.eval()
    client_positions = np.random.rand(params['num_clients'], 2)  # Final client positions
    data = prepare_graph_data(server_positions, client_positions, server_weights, radius)
    cluster_assignments = model(data.x, data.edge_index).argmax(dim=1)
    client_clusters = cluster_assignments[:params['num_clients']]  # Client cluster assignments



    solutions = []
    for cluster_id in range(num_clusters):
        client_indices = np.where(client_clusters == cluster_id)[0]
        client_demands_cluster = client_demands[client_indices, :]
        # solution = fractional_linear_programming(params['num_servers'], params['num_functions'], params['num_clients'], server_weights, radius, client_positions, server_positions, client_demands)  # solve the LP problem
        solutions.append(solution)
    final_solution = aggregate_solutions(solutions, params['num_servers'], params['num_clients'])
    return final_solution

# Cluster Assignments:
# The GNN model produces a tensor where each row corresponds to a node (either a client or a server) and each column corresponds to a cluster.
# The values in this tensor represent the logits (unnormalized scores) for each cluster.
# By applying argmax to these logits along the columns, we can determine the most likely cluster for each node.
# This gives us the cluster assignments for both clients and servers.
# Using the Output Client and Server Cluster Assignments:
# Client Clusters: The cluster assignments for the clients are used to partition the clients into different sub-problems, each corresponding to a cluster.
# Server Clusters: The cluster assignments for the servers help in understanding the groupings of servers, but primarily, the client clusters are used to solve the sub-problems.
# Solving LP for Clusters:
# For each client cluster, a smaller LP problem is formulated and solved. This LP problem considers only the clients within the cluster and their demands.
# The server capacities and positions are taken into account when solving these sub-problems.
# Aggregating Solutions:
# The solutions from the individual LP problems are then aggregated to form the final solution. This aggregation process combines the results
# from each cluster to ensure that all client demands are met in an optimal manner.