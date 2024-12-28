from casadi import DM,MX,Function, exp, if_else
import casadi as cs
import os
from neural_network import NeuralNetwork
import torch.nn as nn
import math
#from A3_template.train import create_casadi_function
import numpy as np

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

    # Create the CasADi function
    nn_func = Function('nn_func', [state], [sigmoid_output])  # CasADi function
    
    return nn_func

# LOAD THE NN AND TRANSFORM IT IN A CASADI FUNCTION
A3_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(A3_dir, "nn_models")
input_size = 4  # Adjust this to match your neural network's input size

# Load the CasADi function
back_reach_set_fun = create_casadi_function(robot_name="double_pendulum", NN_DIR=model_dir, input_size=input_size)

# Declare a symbolic variable for differentiation
dummy_input = MX.sym('x', 4)  # Symbolic variable with 4 elements

# Compute the Jacobian of the neural network function
J = cs.jacobian(back_reach_set_fun(dummy_input), dummy_input)

# Create a CasADi function to evaluate the Jacobian numerically
J_func = cs.Function('J_func', [dummy_input], [J])

# Compute the Hessian of the neural network function
H, grad = cs.hessian(back_reach_set_fun(dummy_input), dummy_input)

# Create a CasADi function to evaluate the Hessian numerically
H_func = cs.Function('H_func', [dummy_input], [H])

# Evaluate the Jacobian at a specific input
numerical_input = DM([0.0, 0.0, 25.0, 50.0])
J_evaluated = J_func(numerical_input)
H_evaluated = H_func(numerical_input)

print("Jacobian at input:", J_evaluated)
print("Hessian at input:", H_evaluated)

print("output probability ",back_reach_set_fun(numerical_input))

# opti = cs.Opti()

# opti.solver('ipopt', {'hessian_approximation': 'limited-memory',
#                       'print_level': 5})

# x = opti.variable(4)

# opti.set_initial(x, DM([0, 0, 0, 0])) 

# opti.minimize(-(x[2]+x[3]) + 10 * back_reach_set_fun(x))
# # opti.subject_to(back_reach_set_fun(x) >= 0.5)

# opti.solver('ipopt')

# try:
#     sol = opti.solve()
#     print("Solution x:", sol.value(x))
#     print("Constraint value:", back_reach_set_fun(sol.value(x)))
    
# except RuntimeError as e:
#     print("Solver failed:", e)
#     print("Debugging initial state...")
#     print("Initial x:", opti.debug.value(x))
#     print("Initial constraint value:", opti.debug.value(back_reach_set_fun(x)))