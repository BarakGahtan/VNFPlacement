import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Define a simple LP problem with constraints
n = 5  # Number of variables
x = cp.Variable(n)
A = cp.Parameter((3, n))  # Equality constraint matrix
b = cp.Parameter(3)       # Equality constraint vector
G = cp.Parameter((2*n, n)) # Inequality constraint matrix
h = cp.Parameter(2*n)     # Inequality constraint vector
c = cp.Parameter(n)       # Objective function coefficients

# Define the LP: minimize c^T x subject to Ax = b and Gx <= h
objective = cp.Minimize(c @ x)
constraints = [A @ x == b, G @ x <= h]
problem = cp.Problem(objective, constraints)

# Convert to a differentiable layer
cvxpylayer = CvxpyLayer(problem, parameters=[c, A, b, G, h], variables=[x])

# Example usage in a PyTorch model
class ExampleModel(torch.nn.Module):
    def forward(self, c, A, b, G, h):
        solution, = cvxpylayer(c, A, b, G, h)
        return solution

# Generate random data for c, A, b, G, and h
c_torch = torch.randn(n, requires_grad=True)
A_torch = torch.randn(3, n, requires_grad=True)
b_torch = torch.randn(3, requires_grad=True)
G_torch = torch.randn(2*n, n, requires_grad=True)
h_torch = torch.randn(2*n, requires_grad=True)

# Instantiate the model and perform a forward pass
model = ExampleModel()
solution = model(c_torch, A_torch, b_torch, G_torch, h_torch)

# Compute gradients with respect to inputs
solution.sum().backward()

print("Gradient of solution with respect to c:", c_torch.grad)
print("Gradient of solution with respect to A:", A_torch.grad)
print("Gradient of solution with respect to G:", G_torch.grad)
