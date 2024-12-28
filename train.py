import os
import numpy as np
import torch
import time  # Import the time module for tracking the elapsed time
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
from neural_network import NeuralNetwork
from A3_template.BwRS import is_in_BwRS
from example_robot_data.robots_loader import load
import multiprocessing  # Use native Python multiprocessing module
import l4casadi as l4c
from casadi import MX, Function, exp, if_else

# GLOBAL COMPUTE_LABEL FUNCTION (moved out of generate_data)
def compute_label(state, robot, N, dt, q_bound, tau_bound):
    """
    Computes the label for a given state (whether in BwRS or not).
    
    Parameters:
    - state: A single state vector [position1, position2, velocity1, velocity2]
    - robot: The robot model to use
    - N, dt, q_bound, tau_bound: Parameters for the BwRS computation
    
    Returns:
    - Label indicating whether the state is inside the boundary of the reachable set (BwRS)
    """
    return is_in_BwRS(robot, state, N, dt, q_bound, tau_bound)

# DATA GENERATION FUNCTION WITH TIMER
def generate_data(save_path, num_samples=500, N=25, dt=0.01, use_multiprocessing=False):
    """
    Generates random data for a double pendulum robot's state (positions and velocities),
    computes corresponding labels, and saves them as a torch dataset.
    
    Parameters:
    - save_path: Where the dataset will be saved
    - num_samples: Number of data samples to generate
    - N: Parameter used in the is_in_BwRS computation
    - dt: Time step used in the robot simulation
    - use_multiprocessing: Flag to enable multiprocessing for label computation
    """
    # Start timer for data generation
    start_time = time.time()  # Record the start time
    
    # Define bounds for the robot's state (positions, velocities) and torques
    def create_bounds():
        q_limit = 2.0 * np.pi  # Joint position limit
        v_limit = 10.0         # Joint velocity limit
        torque_limit = 10.0    # Torque limit
        scaling_factor = 0.5   # Scaling factor for torque limits
        
        # State bounds: [theta1, theta2, v1, v2, -theta1, -theta2, -v1, -v2]
        q_bound = [-q_limit, -q_limit, -v_limit, -v_limit, q_limit, q_limit, v_limit, v_limit]
        
        # Torque bounds
        tau_bound = [-torque_limit * scaling_factor, -torque_limit * scaling_factor,
                     torque_limit * scaling_factor, torque_limit * scaling_factor]
        
        return np.array(q_bound), np.array(tau_bound)
    
    q_bound, tau_bound = create_bounds()

    # Load the robot model
    robot = load("double_pendulum")
    
    # Generate random states (positions and velocities)
    pos = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, size=(num_samples, 2))
    vel = np.random.uniform(-30.0, 30.0, size=(num_samples, 2))
    states = np.hstack((pos, vel))  # Combine position and velocity into states
    
    # Parallelized label computation using multiprocessing or single-threaded option
    if use_multiprocessing:
        # Using Python's built-in multiprocessing Pool to compute labels in parallel
        with multiprocessing.Pool() as pool:
            labels = pool.starmap(compute_label, [(state, robot, N, dt, q_bound, tau_bound) for state in states])
    else:
        # Without multiprocessing, compute labels sequentially
        labels = [compute_label(state, robot, N, dt, q_bound, tau_bound) for state in states]
    
    # Convert states and labels to torch tensors
    dataset = {
        "states": torch.tensor(states, dtype=torch.float32),  # States tensor
        "labels": torch.tensor(labels, dtype=torch.float32)   # Labels tensor
    }
    
    # Save the dataset as a PyTorch .pt file
    torch.save(dataset, save_path)
    
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    # Print how long the data generation took
    print(f"Dataset generated and saved to {save_path}")
    print(f"Time taken for dataset generation: {elapsed_time:.2f} seconds")

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

# MODEL TRAINING FUNCTION
def train_model(dataset_path, model_save_dir):
    """
    Trains the neural network model on the dataset.
    
    Parameters:
    - dataset_path: Path to the dataset file
    - model_save_dir: Directory where the trained model will be saved
    """
    print("Training model...")
    
    # Load the dataset from file
    dataset = torch.load(dataset_path)
    states = dataset["states"]
    labels = dataset["labels"].unsqueeze(1)  # Make labels 2D for PyTorch [batch_size, 1]
    
    # Split the dataset into training and testing subsets
    full_dataset = TensorDataset(states, labels)  # Combine states and labels
    train_size = int(0.8 * len(full_dataset))     # 80% for training
    test_size = len(full_dataset) - train_size    # Remaining 20% for testing
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create DataLoader instances for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define a simple neural network model
    input_size = 4  # 2 joint positions + 2 joint velocities
    hidden_size = 128  # Hidden layer size
    output_size = 1  # Binary classification output
    
    # Initialize the model with a tanh activation function
    model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, 
                          output_size=output_size, activation=nn.Tanh())
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    
    # Set the device for training (CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Train the model for a number of epochs
    num_epochs = 150
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU if available
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()  # Reset gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update model weights
            
            total_loss += loss.item()
        
        # Print the loss after each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    
    # Validate model after training
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Compute model outputs and predictions
            outputs = torch.sigmoid(model(X_batch))  # Apply sigmoid to output logits
            predictions = (outputs > 0.5).float()    # Convert logits to binary predictions
            total += y_batch.size(0)
            correct += (predictions == y_batch).sum().item()  # Count correct predictions
    
    # Print test accuracy
    print(f"Test Accuracy: {correct / total:.4f}")
    
    # Save the trained model to disk
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)  # Create directory if not exists
    
    model_path = os.path.join(model_save_dir, "model.pt")
    torch.save({'model': model.state_dict()}, model_path)
    print(f"Model saved to {model_path}")
    
    # Save the model as a CasADi function using l4casadi
    nn_func = create_casadi_function(robot_name="double_pendulum", NN_DIR=model_save_dir, input_size=input_size)
    print("CasADi function created.")

# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    # Paths for saving/loading data and models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset", "training_data.pt")
    model_save_dir = os.path.join(script_dir, "nn_models")
    
    # Create directories if not exist
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Parameters for data generation
    num_samples = 1000  # Number of samples to generate
    N = 25              # Parameter for BwRS computation
    dt = 0.01           # Time step for simulation
    
    # Flag to enable multiprocessing (True or False)
    use_multiprocessing = False  # Set to True to enable parallel label computation

    # Generate data and train model
    generate_data(dataset_path, num_samples=num_samples, N=N, dt=dt, use_multiprocessing=use_multiprocessing)
    train_model(dataset_path, model_save_dir)



