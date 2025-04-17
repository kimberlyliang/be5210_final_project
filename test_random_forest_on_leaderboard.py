"""
Test Script: Random Forest on Leaderboard Data
This script loads leaderboard ECoG and dataglove data, trains a Random Forest Regressor,
generates predictions, and saves the output as predictions.mat for testing purposes.
"""

import numpy as np
import scipy.io as sio
from scipy import signal as sig
from scipy.io import savemat
import pickle

# Constants
FS = 1000  # Sampling rate
WINDOW_LENGTH = 0.1  # seconds
WINDOW_OVERLAP = 0.05  # seconds
N_WIND = 3  # Number of windows for R matrix

def get_windowed_feats(raw_ecog, fs=FS, window_length=WINDOW_LENGTH, window_overlap=WINDOW_OVERLAP):
    """Process data through filtering and feature calculation."""
    window_length = int(window_length * fs)
    window_overlap = int(window_overlap * fs)
    step = window_length - window_overlap
    num_samples, _ = raw_ecog.shape
    starts = np.arange(0, num_samples - window_length + 1, step)

    all_feats = []
    for start in starts:
        end = start + window_length
        window = raw_ecog[start:end, :]
        all_feats.append(window.flatten())

    return np.array(all_feats)

def create_R_matrix(features, N_wind=N_WIND):
    """Create the R matrix for linear decoding."""
    M, D = features.shape
    R = np.zeros((M, N_wind * D))

    for i in range(M):
        window = features[i:i + N_wind, :]
        R[i, :] = window.flatten(order='C')
    
    bias = np.ones((M, 1))
    R = np.hstack((bias, R))
    return R

# Load the leaderboard data
test_file = 'leaderboard_data.mat'
test_data = sio.loadmat(test_file)
leaderboard_ecog = test_data['leaderboard_ecog']

# Process the leaderboard data
leaderboard_feats = [get_windowed_feats(data[0]) for data in leaderboard_ecog]
R_leaderboard = [create_R_matrix(feats) for feats in leaderboard_feats]

# Load your trained R matrix (algorithm.pkl)
with open('algorithm.pkl', 'rb') as f:
    f = pickle.load(f)  # This should be your trained filter matrix

# Make predictions
all_predictions = []
for subject_data in R_leaderboard:
    subject_predictions = subject_data @ f
    all_predictions.append(subject_predictions)

# Save predictions
predicted_dg = np.array(all_predictions, dtype=object)
savemat('predictions.mat', {'predicted_dg': predicted_dg})
