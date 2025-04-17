import numpy as np
import scipy.io as sio
from scipy import signal as sig
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import pickle
from scipy.io import savemat

# Constants
FS = 1000  # Sampling rate
WINDOW_LENGTH = 0.1  # seconds
WINDOW_OVERLAP = 0.05  # seconds
N_WIND = 3  # Number of windows for R matrix

def bandpower(data, fs, fmin, fmax):
    """Calculate the power in a frequency band using Welch's method."""
    f, Pxx = sig.welch(data, fs=fs, nperseg=min(256, len(data)), nfft=1024)
    band = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[band], f[band])

def filter_data(raw_eeg, fs=FS):
    """Filter the raw ECoG data."""
    def notch_filter(data, freq, fs, Q=30):
        b, a = sig.iirnotch(w0=freq/(fs/2), Q=Q)
        return sig.filtfilt(b, a, data, axis=0)

    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = sig.butter(order, [low, high], btype='band')
        return sig.filtfilt(b, a, data, axis=0)

    # Apply notch filters
    filtered = raw_eeg.copy()
    for freq in [60, 120, 180, 240, 300]:
        filtered = notch_filter(filtered, freq, fs)

    # Apply bandpass filter
    clean_data = bandpass_filter(filtered, lowcut=1, highcut=200, fs=fs)
    return clean_data

def get_features(filtered_window, fs=FS):
    """Extract features from a window of filtered data."""
    window_samples, num_channels = filtered_window.shape
    num_features = 6
    features = np.zeros((num_channels, num_features))

    for ch in range(num_channels):
        signal = filtered_window[:, ch]
        features[ch, 0] = np.mean(signal)
        features[ch, 1] = bandpower(signal, fs, 5, 15)    # Mu band
        features[ch, 2] = bandpower(signal, fs, 20, 25)   # Beta band
        features[ch, 3] = bandpower(signal, fs, 75, 115)  # High gamma
        features[ch, 4] = bandpower(signal, fs, 125, 160) # Upper high gamma
        features[ch, 5] = bandpower(signal, fs, 160, 175)
    return features

def get_windowed_feats(raw_ecog, fs=FS, window_length=WINDOW_LENGTH, window_overlap=WINDOW_OVERLAP):
    """Process data through filtering and feature calculation."""
    # Handle the nested array structure
    if isinstance(raw_ecog, np.ndarray) and len(raw_ecog.shape) == 1:
        raw_ecog = raw_ecog[0]  # Unwrap the array
        
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

def compute_linear_filter(R, Y):
    """Compute the optimal linear filter."""
    alpha = 0  # Regularization parameter
    RTR = R.T @ R + alpha * np.eye(R.shape[1])
    RTY = R.T @ Y
    return np.linalg.solve(RTR, RTY)

def train_models(R_train, train_glove):
    """Train all models on the training data."""
    models = {}
    
    # Linear Filter
    f = compute_linear_filter(R_train, train_glove)
    models['linear'] = f
    
    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(R_train, train_glove)
    models['ridge'] = ridge_model
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(R_train, train_glove)
    models['rf'] = rf_model
    
    # MLP
    mlp_model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                           solver='adam', max_iter=500, random_state=42)
    mlp_model.fit(R_train, train_glove)
    models['mlp'] = mlp_model
    
    return models

def evaluate_models(models, R_test, test_glove):
    """Evaluate all models on test data."""
    results = {}
    for name, model in models.items():
        if name == 'linear':
            Y_pred = R_test @ model
        else:
            Y_pred = model.predict(R_test)
        
        correlations = []
        for finger_idx in range(5):
            r, _ = pearsonr(Y_pred[:, finger_idx], test_glove[:, finger_idx])
            correlations.append(r)
        
        results[name] = correlations
    return results

class FingerMovementPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
    def train(self, train_ecog, train_glove):
        """Train the model on training data."""
        # Process training data
        train_feats = [get_windowed_feats(data) for data in train_ecog]
        R_train = [create_R_matrix(feats) for feats in train_feats]
        
        # Train on all subjects
        X_train = np.vstack(R_train)
        y_train = np.vstack(train_glove)
        self.model.fit(X_train, y_train)
        
    def predict(self, leaderboard_ecog):
        """Make predictions on leaderboard data."""
        # Process leaderboard data
        leaderboard_feats = [get_windowed_feats(data[0]) for data in leaderboard_ecog]
        R_leaderboard = [create_R_matrix(feats) for feats in leaderboard_feats]
        
        # Make predictions
        all_predictions = []
        for subject_data in R_leaderboard:
            subject_predictions = self.model.predict(subject_data)
            all_predictions.append(subject_predictions)
            
        return np.array(all_predictions, dtype=object)

# Create and train the predictor
predictor = FingerMovementPredictor()

# Load training data
train_file = 'raw_training_data.mat'
train_data = sio.loadmat(train_file)
ecog = train_data['train_ecog']
data_glove = train_data['train_dg']

# Train the model
predictor.train(ecog, data_glove)

# Save the trained predictor
with open('finger_movement_predictor.pkl', 'wb') as f:
    pickle.dump(predictor, f)

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