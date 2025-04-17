"""
BE521 Final Project - Random Forest Submission Script
This script loads a pretrained Random Forest model and applies it to the truetest_data.mat file.
It outputs predictions in predictions.mat, formatted with the variable `predicted_dg`.

Usage Instructions (for TAs on Google Colab):
1. Upload this script, your model file (random_forest.pkl), and truetest_data.mat to Colab.
2. Run this script.
3. It will output predictions.mat containing predicted_dg (3x1 cell array with predictions for each subject).

Make sure your model and any additional parameters (e.g. means/stds) are available in the same directory.
"""

import numpy as np
import scipy.io as sio
import pickle
from sklearn.ensemble import RandomForestRegressor
import os

# ---------------- Load pretrained model ----------------
with open("random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- Load the hidden test set ----------------
mat_contents = sio.loadmat("truetest_data.mat")
truetest_data = mat_contents["truetest_data"]  # should be a 3x1 cell array

# ---------------- Helper: Feature extraction ----------------
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=30, fs=1000, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def get_features(ecog_window):
    return np.concatenate([
        np.mean(ecog_window, axis=0),
        np.std(ecog_window, axis=0),
        np.max(ecog_window, axis=0),
        np.min(ecog_window, axis=0)
    ])

def sliding_window_feats(data, window_size=100, step_size=50):
    num_windows = (data.shape[0] - window_size) // step_size + 1
    feats = np.zeros((num_windows, data.shape[1] * 4))
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        feats[i, :] = get_features(data[start:end])
    return feats

# ---------------- Generate predictions ----------------
predicted_dg = []

for subj_idx in range(3):
    ecog = truetest_data[0, subj_idx]
    filtered = butter_lowpass_filter(ecog)
    feats = sliding_window_feats(filtered)
    
    # Predict using RF model
    preds = model.predict(feats)

    # Upsample predictions to match original length
    pred_upsampled = np.repeat(preds, 50, axis=0)
    pred_upsampled = pred_upsampled[:ecog.shape[0], :]  # trim if overshot

    predicted_dg.append(pred_upsampled)

# ---------------- Save output to .mat ----------------
sio.savemat("predictions.mat", {"predicted_dg": predicted_dg})
print("Saved predictions to predictions.mat")
