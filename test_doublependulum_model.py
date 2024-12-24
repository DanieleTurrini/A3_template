import numpy as np
from casadi import MX, Function
import os

# Set the directory where the model is stored
NN_DIR = "/Users/danieleturrini/orc/A3_template/nn_models/nn_doublependulum/"
robot_name = "doublependulum"
casadi_model_path = f"{NN_DIR}{robot_name}_model.pt"



# # Check if the model file exists
# if not os.path.exists(casadi_model_path):
#     raise FileNotFoundError(f"CasADi model not found at: {casadi_model_path}")

# # Load the CasADi function
# nn_func = Function.load(casadi_model_path)

# # Create a sample input (ensure it matches the expected input size of the neural network)
# # For example, if the input is 4-dimensional (2 joint positions + 2 joint velocities):
# sample_input = np.array([0.1, -0.2, 0.3, -0.4])  # Example state vector
# sample_input = sample_input.reshape(1, -1)  # Ensure input is a 1x4 row vector

# # Evaluate the CasADi function
# input_sym = MX.sym("x", 1, 4)  # Symbolic input
# output_sym = nn_func(input_sym)  # Symbolic output

# # Create a callable function to evaluate numerically
# nn_eval = Function("nn_eval", [input_sym], [output_sym])
# result = nn_eval(sample_input)

# # Print the result
# print(f"Sample input: {sample_input}")
# print(f"Network output: {result.full()}")  # Convert CasADi DM to a NumPy array
