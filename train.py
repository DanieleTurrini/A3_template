import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
import numpy as np
from neural_network import NeuralNetwork
import torch.nn as nn
import os

# Load data
data = np.load("training_data.npz")
states = torch.tensor(data['states'], dtype=torch.float32)
labels = torch.tensor(data['labels'], dtype=torch.float32).unsqueeze(1)

# Split data into training and testing sets
dataset = TensorDataset(states, labels)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% for training
test_size = dataset_size - train_size  # Remaining 20% for testing

# Randomly split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Verify the splits
print(f"Training set size: {len(train_dataset)} samples")
print(f"Testing set size: {len(test_dataset)} samples")

# Define the model
input_size = 4  # 2 joint positions + 2 joint velocities
hidden_size = 64  # Adjust as needed
output_size = 1  # Binary classification (1 or 0)

model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, activation=nn.Tanh())

# TRAIN
# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# VALIDATION
# Evaluate on the test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = torch.sigmoid(model(X_batch))
        predictions = (outputs > 0.5).float()
        total += y_batch.size(0)
        correct += (predictions == y_batch).sum().item()

    print(f"Test Accuracy: {correct / total:.4f}")

# EXPORT THE MODEL WITH L4CASADI
# Save the model weights
NN_DIR = "./nn_models"  # Directory to save the model
if not os.path.exists(NN_DIR):
    os.makedirs(NN_DIR)
torch.save({'model': model.state_dict()}, f"{NN_DIR}model.pt")

# Export the model to CasADi
robot_name = "my_robot"
load_weights = True
model.create_casadi_function(robot_name, NN_DIR, input_size, load_weights)

# The CasADi function can now be used in optimization problems
print("Neural network exported successfully.")