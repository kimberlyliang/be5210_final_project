"""
CNN Training and Prediction for ECoG Finger Movement Prediction
This script:
- Loads raw_training_data.mat and leaderboard_data.mat
- Trains a CNN per subject on overlapping ECoG windows
- Predicts on leaderboard data using the trained models
- Saves predictions to predictions.mat in the correct 3x1 cell array format
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---------------- Model ----------------
class ECoGCNN(nn.Module):
    def __init__(self, input_channels):
        super(ECoGCNN, self).__init__()
        
        # Convolutional layers for spectrogram
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # Adjust based on your spectrogram dimensions
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        self.fc3 = nn.Linear(128, 5)  # 5 fingers

    def forward(self, x):
        # Input shape: (batch_size, channels, freq_bins, time_steps)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

# ---------------- Helpers ----------------
def butter_filter(data, cutoff=30, fs=1000):
    b, a = butter(4, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data, axis=0)

def create_windows(X, y=None, win=200, stride=50):
    num_windows = (X.shape[0] - win) // stride + 1
    X_win = np.zeros((num_windows, X.shape[1], win))
    y_win = np.zeros((num_windows, 5)) if y is not None else None

    for i in range(num_windows):
        start = i * stride
        end = start + win
        X_win[i] = X[start:end].T
        if y is not None:
            y_win[i] = y[end - 1]  # predict at window end
    return X_win, y_win

def compute_spectrogram(data, fs=1000, nperseg=256, noverlap=128):
    """
    Compute spectrogram for each channel
    data: (time_steps, channels)
    returns: (freq_bins, time_steps, channels)
    """
    spectrograms = []
    for channel in range(data.shape[1]):
        f, t, Sxx = sig.spectrogram(data[:, channel], 
                                   fs=fs, 
                                   nperseg=nperseg, 
                                   noverlap=noverlap)
        spectrograms.append(Sxx)
    
    # Stack spectrograms along channel dimension
    spectrograms = np.stack(spectrograms, axis=2)  # (freq_bins, time_steps, channels)
    return spectrograms

# ---------------- Training Function ----------------
def train_model(model, train_loader, val_loader, device, num_epochs=50):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_loss = float('inf')
    patience, counter = 10, 0

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

# ---------------- Main Pipeline ----------------
def main():
    # First, load the training data
    train_file = 'raw_training_data.mat'
    train_data = sio.loadmat(train_file)
    
    # Access the data correctly - it's a 3x1 cell array
    ecog_train = train_data['train_ecog']  # Shape: (3, 1)
    glove_train = train_data['train_dg']   # Shape: (3, 1)
    
    # Print shapes to verify
    print("ECoG data shape:", ecog_train.shape)
    print("Glove data shape:", glove_train.shape)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train a model for each subject
    cnn_models = []
    
    for subj in range(3):
        print(f"\nTraining CNN for subject {subj + 1}")
        
        # Get subject data
        X = ecog_train[subj, 0]  # Raw ECoG data
        y = glove_train[subj, 0]  # Glove data
        
        # Compute spectrograms
        print("Computing spectrograms...")
        spectrograms = compute_spectrogram(X)
        
        # Print shapes to verify
        print(f"Subject {subj + 1} Spectrogram shape:", spectrograms.shape)
        
        # Prepare data for CNN
        X = torch.FloatTensor(spectrograms)  # Shape: (freq_bins, time_steps, channels)
        X = X.permute(2, 0, 1)  # Shape: (channels, freq_bins, time_steps)
        
        # Create train/val split
        train_size = int(0.8 * X.shape[2])  # Use time_steps dimension
        X_train = X[:, :, :train_size]
        X_val = X[:, :, train_size:]
        
        # Downsample y to match spectrogram time steps
        # Calculate the downsampling factor based on spectrogram time steps
        downsampling_factor = X.shape[2] // y.shape[0]
        y = y[::downsampling_factor]
        
        # Ensure y has the correct length
        y = y[:X.shape[2]]  # Trim to match X's time steps
        
        y_train = torch.FloatTensor(y[:train_size])
        y_val = torch.FloatTensor(y[train_size:])
        
        # Print shapes to verify
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train.permute(2, 0, 1), y_train)  # (time_steps, channels, freq_bins)
        val_dataset = TensorDataset(X_val.permute(2, 0, 1), y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Create and train model
        model = ECoGCNN(input_channels=X.shape[0]).to(device)
        train_model(model, train_loader, val_loader, device)
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        cnn_models.append(model)

if __name__ == "__main__":
    main()
