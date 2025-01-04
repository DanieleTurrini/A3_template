import os
import numpy as np
import torch
import time  # Import the time module for tracking the elapsed time
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
from neural_network import NeuralNetwork
from A3_template.BwRS import is_in_BwRS
import multiprocessing  # Use native Python multiprocessing module
import l4casadi as l4c
from casadi import MX, Function, exp, if_else

# GENERATE DATA FUNCTION
def generate_data(save_path, num_samples=500, N=25, use_multiprocessing=False):
    """
    Generates random data for a double pendulum robot's state (positions and velocities),
    computes corresponding labels, and saves them as a torch dataset.
    
    Parameters:
    - save_path: Where the dataset will be saved
    - num_samples: Number of data samples to generate
    - N: Parameter used in the is_in_BwRS computation
    - use_multiprocessing: Flag to enable multiprocessing for label computation
    """
    # Start timer for data generation
    start_time = time.time()
    print(f"Starting data generation for {num_samples} samples...")
    
    # Generate random states (positions and velocities)
    pos_J1 = np.random.uniform(-1.5 * np.pi, 0.2 * np.pi, size=(num_samples, 1))
    pos_J2 = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, size=(num_samples, 1))
    vel_J1 = np.random.uniform(-7.0, 7.0, size=(num_samples, 1))
    vel_J2 = np.random.uniform(-2.0, 2.0, size=(num_samples, 1))
    states = np.hstack((pos_J1, pos_J2, vel_J1, vel_J2))  # Combine position and velocity into states
    
    labels = []  # Initialize an empty list to store labels

    if use_multiprocessing:
        # Using Python's multiprocessing Manager for shared progress tracking
        with multiprocessing.Manager() as manager:
            progress_counter = manager.Value('i', 0)  # Shared counter to track progress
            
            def label_computation(state):
                result = is_in_BwRS(state, N)
                with progress_counter.get_lock():
                    progress_counter.value += 1
                # Print progress
                processed_samples = progress_counter.value
                elapsed_time = time.time() - start_time
                time_per_sample = elapsed_time / processed_samples
                remaining_time = time_per_sample * (num_samples - processed_samples)
                print(f"Progress: {processed_samples}/{num_samples} samples generated. "
                      f"Elapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s", end="\r")
                return result

            with multiprocessing.Pool() as pool:
                labels = pool.map(label_computation, states)
    else:
        # Sequential processing with progress tracking
        batch_size = 100  # Adjust batch size for progress tracking
        for i in range(0, num_samples, batch_size):
            batch_states = states[i:i + batch_size]
            batch_labels = [is_in_BwRS(state, N) for state in batch_states]
            labels.extend(batch_labels)
            
            # Progress tracking
            elapsed_time = time.time() - start_time
            processed_samples = i + batch_size
            processed_samples = min(processed_samples, num_samples)  # Avoid overflow
            
            # Estimate time remaining
            time_per_sample = elapsed_time / processed_samples
            remaining_time = time_per_sample * (num_samples - processed_samples)
            print(f"Progress: {processed_samples}/{num_samples} samples generated. "
                  f"Elapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s", end="\r")
    
    # Convert states and labels to torch tensors
    dataset = {
        "states": torch.tensor(states, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.float32)
    }
    
    # Save the dataset as a PyTorch .pt file
    torch.save(dataset, save_path)
    
    # End timer
    end_time = time.time()
    print(f"\nDataset generated and saved to {save_path}")
    print(f"Time taken for dataset generation: {end_time - start_time:.2f} seconds")
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
        model = NeuralNetwork(input_size=input_size, hidden_size=32, output_size=1, activation=nn.Tanh())
        model.load_state_dict(nn_data['model'])  # Load the trained weights

    state = MX.sym("x", input_size)  # Define CasADi symbolic variable for the state
    print(f"State shape: {state.shape}")

    # Initialize L4CasADi wrapper
    l4c_model = l4c.L4CasADi(model, 
                             device='cuda' if torch.cuda.is_available() else 'cpu',
                             name=f'{robot_name}_model', 
                             build_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "l4casadi"))

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
    hidden_size = 32  # Hidden layer size
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
    num_epochs = 50
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
    
    '''# Save the model as a CasADi function using l4casadi
    nn_func = create_casadi_function(robot_name="double_pendulum", NN_DIR=model_save_dir, input_size=input_size)
    print("CasADi function created.")'''

# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    # Paths for saving/loading data and models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset", "training_data_200.pt")
    model_save_dir = os.path.join(script_dir, "nn_models")
    
    # Create directories if not exist
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Parameters for data generation
    num_samples = 5000  # Number of samples to generate
    N = 200             # Parameter for BwRS computation
    dt = 0.01           # Time step for simulation
    
    # Flag to enable multiprocessing (True or False)
    use_multiprocessing = False  # Set to True to enable parallel label computation

    # Generate data and train model
    generate_data(dataset_path, num_samples=num_samples, N=N, use_multiprocessing=use_multiprocessing)
    generate_data(dataset_path, num_samples=num_samples, N=N, use_multiprocessing=use_multiprocessing)
    train_model(dataset_path, model_save_dir)





