import torch
import torch.nn as nn
import torch.optim as optim

# Define the dummy neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define the Adaptive Weighting Class
class AdaptiveCdistWeighting:
    def __init__(self, size, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.moment1 = torch.zeros(size)
        self.moment2 = torch.zeros(size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update(self, gradients):
        self.t += 1
        self.moment1 = self.beta1 * self.moment1 + (1 - self.beta1) * gradients
        self.moment2 = self.beta2 * self.moment2 + (1 - self.beta2) * (gradients ** 2)
        m1_hat = self.moment1 / (1 - self.beta1 ** self.t)
        m2_hat = self.moment2 / (1 - self.beta2 ** self.t)
        W_epsilon = torch.sqrt(m2_hat) / (m1_hat + self.epsilon)
        return W_epsilon

# Define the PairwiseLoss class
class PairwiseLoss(nn.Module):
    def __init__(self, size):
        super(PairwiseLoss, self).__init__()
        self.adaptive_weighting = AdaptiveCdistWeighting(size)
        self.cached_gradients = torch.zeros(size)

    def forward(self, features, compute_gradients=True):
        cdist_matrix = torch.cdist(features, features)
        if compute_gradients:
            cdist_matrix.retain_grad()
            loss = torch.linalg.norm(cdist_matrix)
            loss.backward(retain_graph=True)
            self.cached_gradients = cdist_matrix.grad

        W_epsilon = self.adaptive_weighting.update(self.cached_gradients)
        balanced_cdist = cdist_matrix * W_epsilon
        return torch.linalg.norm(balanced_cdist)

# Initialize a dummy model and optimizer
input_size = 10
hidden_size = 8
output_size = 10
batch_size = 16

model = SimpleNet(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use the custom loss function
loss_function = PairwiseLoss((batch_size, batch_size))

# Generate random data for training
for epoch in range(5):
    inputs = torch.randn(batch_size, input_size)

    # Forward pass
    outputs = model(inputs)

    # Compute the loss (without re-calculating cdist gradients)
    loss = loss_function(outputs, compute_gradients=(epoch == 0))

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for tracking
    print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

