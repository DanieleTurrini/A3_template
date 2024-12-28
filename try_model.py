from casadi import DM,MX,Function, exp, if_else
import casadi as cs
import os
from neural_network import NeuralNetwork
import torch.nn as nn
import math
from A3_template.train import create_casadi_function
import numpy as np

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

# Evaluate the Jacobian at a specific input
numerical_input = DM([0.0, 0.0, 25.0, 50.0])
J_evaluated = J_func(numerical_input)

print("Jacobian at input:", J_evaluated)

# Evaluate the function itself at the same input
output = back_reach_set_fun(numerical_input)
print("Function output:", output)

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