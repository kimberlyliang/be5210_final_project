import scipy.io
import scipy.interpolate
import scipy.signal
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

pred_data = scipy.io.loadmat("leaderboard_data.mat")
proj_data = scipy.io.loadmat("raw_training_data.mat")

data_glove = proj_data['train_dg']
ecog = proj_data['train_ecog']
train_data_glove = []
test_data_glove = []
train_ecog = []
test_ecog = []

for i in range(len(data_glove)):
    num_samples = data_glove[i][0].shape[0]
    split_idx = int(num_samples * 2/3)
    train_data_glove.append(data_glove[i][0][:split_idx])
    test_data_glove.append(data_glove[i][0][split_idx:])
    train_ecog.append(ecog[i][0][:split_idx])
    test_ecog.append(ecog[i][0][split_idx:])

def generate_sliding_windows(data, window_length_ms, overlap_ms, fs=1000):
    window_length_samples = int(window_length_ms * fs / 1000)
    overlap_samples = int(overlap_ms * fs / 1000)
    step_samples = window_length_samples - overlap_samples

    num_windows = (data.shape[0] - window_length_samples) // step_samples + 1
    feature_windows = np.zeros((num_windows, window_length_samples, data.shape[1]))

    for i in range(num_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_length_samples
        feature_windows[i] = data[start_idx:end_idx]

    return feature_windows

def extract_band_power_features(window, fs=1000):
    features = []
    mean_voltage = np.mean(window, axis=0)
    features.append(mean_voltage)
    nperseg = window.shape[0]
    f, Pxx = scipy.signal.welch(np.transpose(window), fs=fs, nperseg=nperseg)

    bands = [(5, 15), (20, 25), (75, 115), (125, 160), (160, 175)] # Hz
    for low, high in bands:
        mask = (f >= low) & (f <= high)
        band_power = np.mean(Pxx[:, mask], axis=1) if np.any(mask) else np.zeros(Pxx.shape[0])
        features.append(band_power)

    return np.concatenate(features)


# preprocessing
def resample_glove_data(glove_data, num_target_samples, fs=1000):
    original_times = np.arange(len(glove_data)) / fs
    target_times = np.arange(num_target_samples) * 0.05

    interpolated = np.zeros((num_target_samples, glove_data.shape[1]))
    for i in range(glove_data.shape[1]):
        interpolator = scipy.interpolate.interp1d(
            original_times, glove_data[:, i], kind='linear',
            bounds_error=False, fill_value="extrapolate"
        )
        interpolated[:, i] = interpolator(target_times)
    return interpolated

def create_lagged_features(features, targets, lag=3):
    num_samples = len(features)
    X = np.array([features[i : i + lag].flatten() for i in range(num_samples - lag + 1)])
    Y = targets[lag - 1:]
    return X, Y

# prediction model
def train_linear_regression(X, Y):
  XT = np.transpose(X)
  XTXinv = np.linalg.inv(np.matmul(XT, X))
  model_params = np.matmul(np.matmul(XTXinv, XT), Y)
  return model_params

def train_random_forest(X, Y):
  rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
  rf.fit(X, Y)
  return rf

def train_xgboost(X, Y):
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X, Y)
    return xgb

# post-processing functions
def upsample_predictions_spline(predictions, original_freq, target_freq, total_duration):
    num_original_samples = int(total_duration * original_freq)
    num_target_samples = int(total_duration * target_freq)

    if predictions.shape[0] != num_original_samples:
         total_duration = predictions.shape[0] / original_freq
         num_original_samples = predictions.shape[0]
         num_target_samples = int(total_duration * target_freq)


    original_times = np.linspace(0, total_duration, num_original_samples, endpoint=False)
    target_times = np.linspace(0, total_duration, num_target_samples, endpoint=False)

    num_fingers = predictions.shape[1]
    upsampled_predictions = np.zeros((num_target_samples, num_fingers))

    for finger_idx in range(num_fingers):
        interpolator = scipy.interpolate.CubicSpline(original_times, predictions[:, finger_idx], bc_type='natural') # natural is often more stable
        upsampled_predictions[:, finger_idx] = interpolator(target_times)

    return target_times, upsampled_predictions

def pad_array_to_length(data, target_length):
    current_length = data.shape[0]
    if current_length == target_length:
        return data
    elif current_length > target_length:
        return data[:target_length]
    else:
        pad_total = target_length - current_length
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        padding = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, padding, mode='constant', constant_values=0)


def smooth_predictions_moving_average(data, window_size):
    smoothed_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        smoothed_data[:, col] = np.convolve(data[:, col], np.ones(window_size) / window_size, mode='same')
    return smoothed_data

from scipy.stats import pearsonr

def compute_finger_correlations(test_glove_data, predictions):
    correlations = []
    num_fingers = test_glove_data.shape[1]
    if predictions.shape[0] != test_glove_data.shape[0]:
         min_len = min(predictions.shape[0], test_glove_data.shape[0])
         test_glove_data = test_glove_data[:min_len]
         predictions = predictions[:min_len]

    for i in range(num_fingers):
        correlation, _ = pearsonr(test_glove_data[:, i], predictions[:, i])
        correlations.append(correlation)
    return correlations

predicted_dg_list = []
leaderboard_ecog = pred_data['leaderboard_ecog']
trained_models = []
SAMPLING_FREQ = 1000
WINDOW_LEN_MS = 100
OVERLAP_MS = 50
PREDICTION_FREQ = 1 / ( (WINDOW_LEN_MS - OVERLAP_MS) / 1000.0 )
LAG_STEPS = 3
SMOOTHING_WINDOW = 8

for subj_idx in range(len(ecog)):
    subj_ecog_train = ecog[subj_idx][0]
    subj_glove_train = data_glove[subj_idx][0]
    subj_ecog_leaderboard = leaderboard_ecog[subj_idx][0]

    train_windows = generate_sliding_windows(subj_ecog_train, WINDOW_LEN_MS, OVERLAP_MS, SAMPLING_FREQ)
    train_features = np.array([extract_band_power_features(window, SAMPLING_FREQ) for window in train_windows])

    resampled_glove = resample_glove_data(subj_glove_train, len(train_features), SAMPLING_FREQ)

    X_train, Y_train = create_lagged_features(train_features, resampled_glove, LAG_STEPS)
 
    model = train_random_forest(X_train, Y_train)
    trained_models.append(model)

    leaderboard_windows = generate_sliding_windows(subj_ecog_leaderboard, WINDOW_LEN_MS, OVERLAP_MS, SAMPLING_FREQ)
    leaderboard_features = np.array([extract_band_power_features(window, SAMPLING_FREQ) for window in leaderboard_windows])
    if len(leaderboard_features) >= LAG_STEPS:
         X_pred = np.array([leaderboard_features[i:i+LAG_STEPS].flatten() for i in range(len(leaderboard_features) - LAG_STEPS + 1)])
    else:
         target_len = subj_ecog_leaderboard.shape[0]
         num_fingers = subj_glove_train.shape[1]
         padded_predictions = np.zeros((target_len, num_fingers))
         predicted_dg_list.append(padded_predictions)
         continue 

    raw_predictions = model.predict(X_pred)

    smoothed_predictions = smooth_predictions_moving_average(raw_predictions, SMOOTHING_WINDOW)

    target_len = subj_ecog_leaderboard.shape[0]
    total_duration_pred = smoothed_predictions.shape[0] / PREDICTION_FREQ
    _, upsampled_predictions = upsample_predictions_spline(
        smoothed_predictions,
        original_freq=PREDICTION_FREQ,
        target_freq=SAMPLING_FREQ,
        total_duration=total_duration_pred
        )

    final_predictions = pad_array_to_length(upsampled_predictions, target_len)

    predicted_dg_list.append(final_predictions)

# --- Save Results ---

# Save predictions to .mat file in a 1xN cell array format
num_subjects = len(predicted_dg_list)
cell_array_1xN = np.empty((1, num_subjects), dtype=object)
for i, arr in enumerate(predicted_dg_list):
    cell_array_1xN[0, i] = arr

scipy.io.savemat(
    'predicted_glove_data.mat',
    {'predicted_dg': cell_array_1xN},
    format='5'
)
print(f"\nSaved predicted dataglove data ({cell_array_1xN.shape}) to predicted_glove_data.mat")

# Save trained models to .pkl file
trained_models_array = np.array(trained_models, dtype='object')
with open('trained_prediction_models.pkl', 'wb') as f:
    pickle.dump(trained_models_array, f)
print("Saved trained models to trained_prediction_models.pkl")


# --- Evaluate on Test Set (Optional) ---
print("\n--- Evaluating models on the held-out test set ---")
all_subject_correlations = []

for subj_idx in range(len(test_ecog)): # Use the test split ecog
    print(f"\n--- Evaluating Subject {subj_idx + 1} ---")
    subj_ecog_test = test_ecog[subj_idx] # Test ECoG data
    subj_glove_test = test_data_glove[subj_idx] # Corresponding true glove data
    model = trained_models[subj_idx]

    if model is None:
        print("Model was not trained for this subject (possibly due to short leaderboard data). Skipping evaluation.")
        all_subject_correlations.append([np.nan] * 5) # Placeholder for 5 fingers
        continue

    # 1. Feature Extraction (Test Data)
    test_windows = generate_sliding_windows(subj_ecog_test, WINDOW_LEN_MS, OVERLAP_MS, SAMPLING_FREQ)
    test_features = np.array([extract_band_power_features(window, SAMPLING_FREQ) for window in test_windows])
    # print(f"Subject {subj_idx+1} - Test Features: {test_features.shape}")

    # 2. Create Lagged Features (Test Data)
    if len(test_features) >= LAG_STEPS:
        X_test_lagged = np.array([test_features[i:i+LAG_STEPS].flatten() for i in range(len(test_features) - LAG_STEPS + 1)])
        # print(f"Subject {subj_idx+1} - Lagged Test Features (X_test_lagged): {X_test_lagged.shape}")
    else:
        print(f"Test data too short ({len(test_features)} samples) for lag {LAG_STEPS}. Cannot evaluate.")
        all_subject_correlations.append([np.nan] * 5)
        continue

    # 3. Make Predictions (Test Data)
    raw_test_predictions = model.predict(X_test_lagged)
    # print(f"Subject {subj_idx+1} - Raw Test Predictions: {raw_test_predictions.shape}")

    # 4. Smooth Predictions (Test Data)
    smoothed_test_predictions = smooth_predictions_moving_average(raw_test_predictions, SMOOTHING_WINDOW)

    # 5. Upsample Predictions (Test Data)
    target_test_len = subj_ecog_test.shape[0] # Target length is the original test ECoG length
    total_duration_test_pred = smoothed_test_predictions.shape[0] / PREDICTION_FREQ
    _, upsampled_test_predictions = upsample_predictions_spline(
        smoothed_test_predictions, PREDICTION_FREQ, SAMPLING_FREQ, total_duration_test_pred
        )
    # print(f"Subject {subj_idx+1} - Upsampled Test Predictions: {upsampled_test_predictions.shape}")

    # 6. Pad/Truncate to Match Original Test ECoG Length
    final_test_predictions = pad_array_to_length(upsampled_test_predictions, target_test_len)
    # print(f"Subject {subj_idx+1} - Final Padded Test Predictions: {final_test_predictions.shape}")

    # 7. Calculate Correlations
    # Ensure the true glove data also matches the final prediction length
    aligned_glove_test = pad_array_to_length(subj_glove_test, target_test_len)
    # print(f"Subject {subj_idx+1} - Aligned True Test Glove: {aligned_glove_test.shape}")

    correlations = compute_finger_correlations(aligned_glove_test, final_test_predictions)
    all_subject_correlations.append(correlations)
    print(f"Subject {subj_idx+1} - Test Set Correlations: {correlations}")

# Optional: Calculate and print average correlation across subjects
avg_correlations = np.nanmean(all_subject_correlations, axis=0)
print(f"\nAverage Test Set Correlations across subjects: {avg_correlations}")
overall_avg_corr = np.nanmean(avg_correlations)
print(f"Overall Average Test Set Correlation: {overall_avg_corr}")

print("\nProcessing complete.")