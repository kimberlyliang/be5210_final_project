import numpy as np
import scipy.io
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.signal import hilbert
from scipy.stats import pearsonr
import pickle

# Load training and leaderboard data
training_data = scipy.io.loadmat("raw_training_data.mat")
testing_data = scipy.io.loadmat("leaderboard_data.mat")
ecog_signals = training_data['train_ecog']
glove_signals = training_data['train_dg']
leaderboard_ecog = testing_data['leaderboard_ecog']

# Split training/testing
train_ecog, test_ecog, train_glove, test_glove = [], [], [], []
for idx in range(len(glove_signals)):
    split_idx = int(glove_signals[idx][0].shape[0] * 2 / 3)
    train_ecog.append(ecog_signals[idx][0][:split_idx])
    test_ecog.append(ecog_signals[idx][0][split_idx:])
    train_glove.append(glove_signals[idx][0][:split_idx])
    test_glove.append(glove_signals[idx][0][split_idx:])

# Sliding window generation
def create_windows(signal, window_ms=100, overlap_ms=50, fs=1000):
    step = int((window_ms - overlap_ms) * fs / 1000)
    length = int(window_ms * fs / 1000)
    n_windows = (signal.shape[0] - length) // step + 1
    windows = np.zeros((n_windows, length, signal.shape[1]))
    for i in range(n_windows):
        windows[i] = signal[i*step:i*step + length]
    return windows

# Feature extraction
def extract_features(segment, fs=1000):
    feats = [np.mean(segment, axis=0)]
    freqs, psd = scipy.signal.welch(segment.T, fs=fs, nperseg=segment.shape[0])
    bands = [(5,15), (20,25), (75,115), (125,160), (160,175)]
    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        feats.append(np.mean(psd[:, mask], axis=1) if np.any(mask) else np.zeros(psd.shape[0]))
    envelope = np.abs(hilbert(segment, axis=0))
    feats.append(np.mean(envelope, axis=0))
    return np.concatenate(feats)

# Downsample glove data
def resample_glove(signal, target_size, fs=1000):
    orig_times = np.arange(signal.shape[0]) / fs
    new_times = np.arange(target_size) * 0.05
    resized = np.zeros((target_size, signal.shape[1]))
    for ch in range(signal.shape[1]):
        f = scipy.interpolate.interp1d(orig_times, signal[:, ch], fill_value="extrapolate")
        resized[:, ch] = f(new_times)
    return resized

# Lag features
def build_lagged_data(X, Y, lag_steps=3):
    X_lagged = np.array([X[i:i+lag_steps].flatten() for i in range(len(X) - lag_steps + 1)])
    Y_lagged = Y[lag_steps-1:]
    return X_lagged, Y_lagged

# Model training
def train_random_forest(X, Y):
    alpha = RandomForestRegressor(n_estimators=100, random_state=42)
    alpha.fit(X, Y)
    return alpha

def train_xgboost(X, Y):
    alpha = XGBRegressor(n_estimators=100, random_state=42)
    alpha.fit(X, Y)
    return alpha

# Post-processing
def moving_average(predictions, window=8):
    smoothed = np.zeros_like(predictions)
    for i in range(predictions.shape[1]):
        smoothed[:, i] = np.convolve(predictions[:, i], np.ones(window)/window, mode='same')
    return smoothed

def interpolate_and_pad(preds, orig_length, old_freq=20, new_freq=1000):
    duration = preds.shape[0] / old_freq
    old_times = np.linspace(0, duration, preds.shape[0])
    new_times = np.linspace(0, duration, int(duration * new_freq))
    expanded = np.zeros((len(new_times), preds.shape[1]))
    for i in range(preds.shape[1]):
        f = scipy.interpolate.CubicSpline(old_times, preds[:, i], bc_type='clamped')
        expanded[:, i] = f(new_times)
    pad_total = orig_length - expanded.shape[0]
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return np.pad(expanded, ((pad_before, pad_after), (0,0)), mode='constant')

# Evaluate correlation
def evaluate_correlation(y_true, y_pred):
    return [pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]

# Main loop
final_predictions = []
all_alphas = []

for subj_idx in range(len(ecog_signals)):
    train_windows = create_windows(ecog_signals[subj_idx][0])
    pred_windows = create_windows(leaderboard_ecog[subj_idx][0])

    train_feats = np.array([extract_features(win) for win in train_windows])
    pred_feats = np.array([extract_features(win) for win in pred_windows])

    glove_train = resample_glove(glove_signals[subj_idx][0], len(train_feats))

    X_train, Y_train = build_lagged_data(train_feats, glove_train, lag_steps=3)
    X_pred = np.array([pred_feats[i:i+3].flatten() for i in range(len(pred_feats) - 2)])

    alpha = train_random_forest(X_train, Y_train)
    all_alphas.append(alpha)

    preds = alpha.predict(X_pred)
    preds_smoothed = moving_average(preds)

    upsampled_preds = interpolate_and_pad(preds_smoothed, leaderboard_ecog[subj_idx][0].shape[0])
    final_predictions.append(upsampled_preds)

    # Print training performance
    print(f"\nSubject {subj_idx+1} Training Correlations:")
    smoothed_train_preds = moving_average(alpha.predict(X_train))
    corr_values = evaluate_correlation(Y_train, smoothed_train_preds)
    for finger_idx, corr in enumerate(corr_values):
        print(f"Finger {finger_idx+1}: {corr:.4f}")

# Save predicted_dg
cell_output = np.empty((len(final_predictions),), dtype=object)
for idx, preds in enumerate(final_predictions):
    cell_output[idx] = preds

scipy.io.savemat('updated_predictions_with_hilbert_alpha.mat', {'predicted_dg': cell_output})

# Save alphas
with open('trained_alphas.pkl', 'wb') as f:
    pickle.dump(np.array(all_alphas, dtype=object), f)
