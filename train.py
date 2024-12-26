from casadi import DM,Function, exp, if_else
import os
from neural_network import NeuralNetwork
import torch.nn as nn
import math

# CREATE CASADI FUNCTION FROM NEURAL NETWORK
def create_casadi_function(robot_name, NN_DIR, input_size, load_weights=True):
    """
    Creates a CasADi function for the trained neural network using L4CasADi.
    
    Parameters:
    - robot_name: The name of the robot model
    - NN_DIR: Directory containing the trained neural network model (.pt file)
    - input_size: The size of the input to the neural network
    - load_weights: Boolean flag to load weights from the `.pt` file (default: True)
    
    Returns:
    - The CasADi function of the trained neural network
    """
    from casadi import MX, Function
    import l4casadi as l4c
    import torch

    print("Initializing L4CasADi Model...")

    # if load_weights is True, we load the neural-network weights from a ".pt" file
    if load_weights:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nn_name = os.path.join(NN_DIR, 'model.pt')
        nn_data = torch.load(nn_name, map_location=device)
        model = NeuralNetwork(input_size=input_size, hidden_size=128, output_size=1, activation=nn.Tanh())
        model.load_state_dict(nn_data['model'])  # Load the trained weights

    state = MX.sym("x", input_size)  # Define CasADi symbolic variable for the state
    print(f"State shape: {state.shape}")

    # Initialize L4CasADi wrapper
    l4c_model = l4c.L4CasADi(model, 
                             device='cuda' if torch.cuda.is_available() else 'cpu',
                             name=f'{robot_name}_model', 
                             build_dir=f'{NN_DIR}nn_{robot_name}')
    
    print(f"Input to model: {l4c_model(state).shape}")

    nn_model = l4c_model(state)
    
    # Apply sigmoid activation to map logits to probabilities
    sigmoid_output = 1 / (1 + exp(-nn_model))  # Sigmoid function
    binary_output = if_else(sigmoid_output >= 0.5, 1.0, 0.0)  # Threshold to produce 0 or 1

    # Create the CasADi function
    nn_func = Function('nn_func', [state], [binary_output])  # CasADi function
    
    return nn_func

# Define necessary parameters
model_dir = "/Users/danieleturrini/orc/A3_template/nn_models"  # Folder where model.pt is saved
input_size = 4  # Adjust this to match your neural network's input size
robot_name = "double_pendulum"

# Load the CasADi function
# Load the CasADi function
nn_func = create_casadi_function(robot_name="double_pendulum", NN_DIR=model_dir, input_size=input_size)

# Test the function with a sample input
sample_input = DM([-0.5, -0.5, 10.0, 0.0])  # Example test input
probability_output = nn_func(sample_input)

print("Sample Input:", sample_input)
print("Probability Output:", probability_output)

