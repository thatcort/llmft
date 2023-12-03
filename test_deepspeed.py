# Import necessary libraries
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam  # Use the GPU-compatible optimizer

# Import DeepSpeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

# Initialize DeepSpeed
from deepspeed import deepspeed

# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create a sample dataset
# Replace this with your actual data loading code
# For simplicity, we'll create random input and output tensors
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# Move data to GPU
inputs, targets = inputs.cuda(), targets.cuda()

# Create a DataLoader
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model and move it to DeepSpeed
model = SimpleModel().cuda()
model, _, _, _ = deepspeed.initialize(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters())  # Use GPU-compatible optimizer

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in dataloader:
        inputs, targets = batch

        # Move data to GPU
        inputs, targets = inputs.cuda(), targets.cuda()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'simple_model.pth')

