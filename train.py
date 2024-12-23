import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np

# Load data
data = np.load("training_data.npz")
states = torch.tensor(data['states'], dtype=torch.float32)
labels = torch.tensor(data['labels'], dtype=torch.float32).unsqueeze(1)

# Create DataLoader
dataset = TensorDataset(states, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)