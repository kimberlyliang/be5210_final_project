import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
from scipy import signal as sig

class ECoGCNN(nn.Module):
    def __init__(self, input_channels):
        super(ECoGCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        
        # Calculate the size after convolutions
        self._to_linear = None
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8, 512),  # Adjusted based on input size
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        
        self.fc3 = nn.Linear(256, 5)  # 5 fingers

    def forward(self, x):
        # Input shape: (batch_size, channels, time_steps)
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

def train_model(model, train_loader, val_loader, device, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")
# Load training data
train_file = 'raw_training_data.mat'
train_data = sio.loadmat(train_file)
ecog = train_data['train_ecog']
data_glove = train_data['train_dg']

# Train a model for each subject
cnn_models = []

for subject_idx in range(3):
    print(f"\nTraining CNN for subject {subject_idx + 1}")
    
    # Get subject data
    X = ecog[subject_idx][0]  # Raw ECoG data
    y = data_glove[subject_idx][0]  # Glove data
    
    # Reshape data for CNN (batch_size, channels, time_steps)
    X = torch.FloatTensor(X)  # Shape: (time_steps, channels)
    X = X.permute(1, 0)      # Shape: (channels, time_steps)
    
    # Create train/val split
    train_size = int(0.8 * X.shape[1])
    X_train = X[:, :train_size]  # Shape: (channels, train_time_steps)
    X_val = X[:, train_size:]    # Shape: (channels, val_time_steps)
    
    # Reshape y to match X
    y = torch.FloatTensor(y)
    y_train = y[:train_size]     # Shape: (train_time_steps, 5)
    y_val = y[train_size:]       # Shape: (val_time_steps, 5)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train.T.unsqueeze(0), y_train)  # Add batch dimension
    val_dataset = TensorDataset(X_val.T.unsqueeze(0), y_val)        # Add batch dimension
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECoGCNN(input_channels=X.shape[0]).to(device)
    train_model(model, train_loader, val_loader, device)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    cnn_models.append(model)

# Load leaderboard data
test_file = 'leaderboard_data.mat'
test_data = sio.loadmat(test_file)
leaderboard_ecog = test_data['leaderboard_ecog']

# Make predictions on leaderboard data
all_predictions = []

for subject_idx in range(3):
    print(f"\nMaking predictions for subject {subject_idx + 1}")
    
    # Get leaderboard data for this subject
    subject_data = leaderboard_ecog[subject_idx][0]
    
    # Prepare data for prediction
    subject_data = torch.FloatTensor(subject_data)  # Shape: (time_steps, channels)
    subject_data = subject_data.permute(1, 0)       # Shape: (channels, time_steps)
    subject_data = subject_data.unsqueeze(0)        # Add batch dimension
    
    # Make predictions
    model = cnn_models[subject_idx]
    model.eval()
    with torch.no_grad():
        subject_predictions = model(subject_data)
    
    # Convert to numpy and ensure correct shape
    subject_predictions = subject_predictions.cpu().numpy()
    if subject_predictions.shape != (147500, 5):
        print(f"Warning: Subject {subject_idx + 1} predictions shape is {subject_predictions.shape}, expected (147500, 5)")
    
    all_predictions.append(subject_predictions)

# Save predictions
predicted_dg = np.array(all_predictions, dtype=object).reshape(3, 1)
sio.savemat('predictions.mat', {'predicted_dg': predicted_dg})