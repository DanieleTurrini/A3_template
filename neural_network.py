import torch
import torch.nn as nn
from casadi import MX, Function
import l4casadi as l4c

class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(NeuralNetwork, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size)
        )
        # Assuming ub is a multiplier, it's better to initialize ub as a parameter and put it on the same device.
        self.ub = torch.ones((output_size, 1))  # Default value, assuming it's a fixed multiplier for the output

    def forward(self, x):
        # Move self.ub to the same device as x
        self.ub = self.ub.to(x.device)
        
        # Ensure input has the correct shape
        if x.ndimension() == 1:
            x = x.view(1, -1)  # Reshape for a single sample
        elif x.ndimension() == 2 and x.shape[0] == 4:
            x = x.T  # Reshaping (4, 1) to (1, 4)

        # Forward pass
        out = self.linear_stack(x) * self.ub  # Multiply with ub (now on the same device)
        return out

    def initialize_weights(self):
        """ Initialize the weights of each layer using Xavier normal initialization. """
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias) 