import os
import torch
import numpy as np
from neural_network import NeuralNetwork
from casadi import MX, DM, Function, exp
from A3_template.BwRS import is_in_BwRS
from A3_template.train import create_casadi_function
import l4casadi as l4c

def compare_real_and_nn(robot_name, nn_model_dir, input_size, x0, N):
    """
    Compares outputs from the real function and the neural network model.

    Parameters:
    - robot_name: Name of the robot, e.g., "double_pendulum".
    - nn_model_dir: Path to the directory with the neural network model.
    - input_size: Size of the input state vector.
    - x0: Initial state vector.
    - N: Number of steps for comparison.
    """
    # Create CasADi neural network function
    nn_func = create_casadi_function(robot_name, nn_model_dir, input_size)



    # Loop to compare the outputs
    for i in range(N):
        # Real function output (1 if feasible, 0 otherwise)
        real_output = is_in_BwRS(x0, 20)

        # Neural network function output
        nn_output = nn_func(DM(x0))
        
        # Print outputs for comparison
        print(f"Step {i + 1}:", "state:", x0)
        print(f"  Real function output: {real_output}")
        print(f"  Neural network output: {float(nn_output)}")
        print()

        # Update the state `x0` (assuming no dynamics in the real function for this example)
        #x0 = np.array(x0) * 0.95  # Example: decay state for next step
        pos1 = np.random.uniform(-1.0 * np.pi, 0.1 * np.pi, size=(1, 1))
        pos2 = np.random.uniform(-0.1 * np.pi, 0.1 * np.pi, size=(1, 1))
        vel = np.random.uniform(-1.0, 1.0, size=(1, 2))
        x0 = np.hstack((pos1,pos2, vel))


# Main setup
if __name__ == "__main__":
    A3_dir = os.path.dirname(os.path.abspath(__file__))
    nn_model_dir = os.path.join(A3_dir, "nn_models")
    robot_name = "double_pendulum"
    input_size = 4
    x0 = [0.0, 0.0, 500.0, 500.0]  # Example starting state
    N = 100  # Number of steps for comparison

    compare_real_and_nn(robot_name, nn_model_dir, input_size, x0, N)
