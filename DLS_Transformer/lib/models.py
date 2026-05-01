import importlib.util
import warnings

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch.nn") is None:
    warnings.warn(
        "PyTorch is not available in this environment. "
        "Install it with: pip install torch",
        ImportWarning,
        stacklevel=2,
    )
else:
    import torch
    import torch.nn as nn
import math
from functools import partial
from datetime import datetime
import os
import numpy as np
import time
from tqdm import tqdm
import copy
import pickle

## ==================================== Positional Encoding ======================================
# Positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

## ====================================== Transformer ============================================
# Define the Transformer Encoder model
class TransformerEncoderModel(nn.Module):
    def __init__(self, time_lag, input_dim, d_model=256, nhead=4, num_layers=4, dropout=0.0):
        super(TransformerEncoderModel, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_len=time_lag, dropout=dropout)

        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)
            
        x = self.fc(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, time_lag, input_dim, hidden_dim=256, num_layers=4, dropout=0.0):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, states=None, return_state=False):
        lstm_out, states = self.lstm(x, states)
        x = self.fc(lstm_out)
        if return_state:
            return x, states
        else:
            return x
    
def make_dataloader(X, Y, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the dataset.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset


    # Create a TensorDataset
    dataset = TensorDataset(X, Y)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
    
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, patience, device, model_dir, data_name, checkpointing=True):
    """
    Train the model with the given parameters.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
        patience (int): Early stopping patience.
        device (torch.device): Device to use for training (CPU or GPU).
        model_dir (str): Directory to save the model.
        data_name (str): Name of the dataset for saving the model.

    Returns:
        dict: A dictionary containing training and test losses.
    """

    best_test_loss = float('inf')
    early_stop_counter = 0
    losses = []
    test_losses = []

    # Generate a timestamp for saving the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(model_dir, f'model_{timestamp}.pth')
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Training loop
    start_time = time.time()
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        ## --------------------------------------- Train ---------------------------------------
        for inputs, targets in train_loader: 
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            total_loss = 0.0

            # Forward pass
            outputs = model(inputs)  # shape: [B, input_dim]
            loss = criterion(outputs, targets)

            # Backward and optimization for current step only
            total_loss += loss
            loss.backward()

            epoch_loss += total_loss.item()
            optimizer.step()

        losses.append(epoch_loss / len(train_loader))

        ## --------------------------------------- Test ---------------------------------------
        # Evaluate the model on the test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        test_losses.append(test_loss / len(test_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

        ## ------------------------------- Early stop and Checkpoint -------------------------------
        # Early stopping and saving the best model
        if epoch > 0:
            if np.isnan(test_losses[-1]) or np.isnan(losses[-1]):
                print(f'NaN loss at epoch {epoch+1}. Stopping training.')
                model.load_state_dict(best_model)
                break
            elif test_loss / len(test_loader) < best_test_loss:
                best_test_loss = test_loss / len(test_loader)
                best_model = copy.deepcopy(model.state_dict())

                # Save the best model checkpoint
                if checkpointing:
                    checkpoint_path = os.path.join(model_dir, f'{data_name}_best_model.pth')
                    torch.save(best_model, checkpoint_path)
                best_epoch = epoch + 1
                print(f'Best model saved at epoch {best_epoch} with test loss: {best_test_loss:.4f}')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    model.load_state_dict(best_model)
                    print(f'Best model loaded from epoch {best_epoch}, with test loss: {best_test_loss:.4f}')
                    break

    end_time = time.time()
    print('Time taken for training: ', end_time - start_time)
    print('Time taken per epoch: ', (end_time - start_time) / num_epochs)

    # Save the final model after training
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Final model saved to {model_save_path}")

    return {"train_losses": losses, "test_losses": test_losses}


def forecastClosedLoop(net_in, A_norm, nt_fore, device):
    """
    Helper: advance LSTM or transformer in closed loop for nt_fore steps
    """
    if hasattr(net_in, 'lstm'):
        X0 = A_norm[:-1]  # [T x n_modes]

        # convert to torch tensor and add batch dimension, then move to device
        X0 = torch.from_numpy(X0).float().unsqueeze(0)  # [1 x T x n_modes]
        X0 = X0.to(device)
        net_in.eval()  # set model to evaluation mode
        
        # if net_in contains an LSTM, we need to get the initial hidden states
        # check if model has lstm layer
    
        _, states = net_in(X0, states=None, return_state=True)
        y_prev, states = net_in(X0[0,-1].unsqueeze(0), states=None, return_state=True)

        n_modes = A_norm.shape[-1]
        A_fore = torch.zeros((1, nt_fore, n_modes)).to(device)

        A_fore[0, 0, :] = y_prev.cpu()

        
        for t in range(1, nt_fore):
            y_prev, states = net_in(y_prev, states, return_state=True)
            A_fore[0, t, :] = y_prev.cpu()  # store the forecasted point

    elif hasattr(net_in, 'encoder_layers'):
        X0 = A_norm[:-1]  # [T x n_modes]

        # convert to torch tensor and add batch dimension, then move to device
        X0 = torch.from_numpy(X0).float()  # [1 x T x n_modes]
        X0 = X0.to(device)
        net_in.eval()  # set model to evaluation mode
        y_prev = X0.unsqueeze(1)
        # print(f"Initial input shape: {y_prev.shape}")  # Debugging print statement
        y_prev = net_in(y_prev)

        n_modes = A_norm.shape[-1]
        A_fore = torch.zeros((1, nt_fore, n_modes)).to(device)
        # print(f"y_prev shape after first forward pass: {y_prev.shape}")  # Debugging print statement
        A_fore[0, 0, :] = y_prev[-1,-1, :].cpu().squeeze()  # store the first forecasted point

        for t in range(1, nt_fore):
            # print(f"y_prev shape: {y_prev.shape}")  # Debugging print statement 
            y_prev = net_in(y_prev[:,-1,:])
            A_fore[0, t, :] = y_prev[:,-1,:].cpu()  # store the forecasted point

    else:
        raise ValueError("Model must have either 'lstm' or 'encoder_layers' attribute")


    return A_fore
