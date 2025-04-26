import scipy.io
import scipy.interpolate
import scipy.signal
import numpy as np
import pickle
import os

SAMPLING_FREQ = 1000
WINDOW_LEN_MS = 100
OVERLAP_MS = 50
PREDICTION_FREQ = 1 / ( (WINDOW_LEN_MS - OVERLAP_MS) / 1000.0 )
LAG_STEPS = 3
SMOOTHING_WINDOW = 8
MODEL_FILENAME = 'trained_prediction_models.pkl'
TEST_DATA_FILENAME = "truetest_data.mat"
OUTPUT_FILENAME = 'predicted_dg.mat'


def generate_sliding_windows(data, window_length_ms, overlap_ms, fs=1000):
    window_length_samples = int(window_length_ms * fs / 1000)
    overlap_samples = int(overlap_ms * fs / 1000)
    step_samples = window_length_samples - overlap_samples

    if data.shape[0] < window_length_samples:
         return np.empty((0, window_length_samples, data.shape[1]))

    num_windows = (data.shape[0] - window_length_samples) // step_samples + 1
    feature_windows = np.zeros((num_windows, window_length_samples, data.shape[1]))

    for i in range(num_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_length_samples
        feature_windows[i] = data[start_idx:end_idx]

    return feature_windows

def extract_band_power_features(window, fs=1000):
    features = []
    # Mean voltage
    mean_voltage = np.mean(window, axis=0)
    features.append(mean_voltage)
    # Band power using Welch's method
    nperseg = window.shape[0]
    f, Pxx = scipy.signal.welch(np.transpose(window), fs=fs, nperseg=nperseg)
    bands = [(5, 15), (20, 25), (75, 115), (125, 160), (160, 175)] # Hz
    for low, high in bands:
        mask = (f >= low) & (f <= high)
        band_power = np.mean(Pxx[:, mask], axis=1) if np.any(mask) else np.zeros(Pxx.shape[0])
        features.append(band_power)
    return np.concatenate(features)


def upsample_predictions_spline(predictions, original_freq, target_freq, total_duration):
    num_original_samples = int(round(total_duration * original_freq))
    num_target_samples = int(round(total_duration * target_freq))


    if predictions.shape[0] == 0:
        return np.linspace(0, total_duration, num_target_samples, endpoint=False), \
               np.zeros((num_target_samples, 5))

    if predictions.shape[0] != num_original_samples:
         num_original_samples = predictions.shape[0]
         total_duration = num_original_samples / original_freq
         num_target_samples = int(round(total_duration * target_freq))

    original_times = np.linspace(0, total_duration, num_original_samples, endpoint=False)
    target_times = np.linspace(0, total_duration, num_target_samples, endpoint=False)

    num_fingers = predictions.shape[1]
    upsampled_predictions = np.zeros((num_target_samples, num_fingers))

    for finger_idx in range(num_fingers):
        interpolator = scipy.interpolate.CubicSpline(original_times, predictions[:, finger_idx], bc_type='natural')
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
    if data.shape[0] == 0:
        return data
    smoothed_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        smoothed_data[:, col] = np.convolve(data[:, col], np.ones(window_size) / window_size, mode='same')
    return smoothed_data

if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILENAME}")
with open(MODEL_FILENAME, 'rb') as f:
    trained_models = pickle.load(f)
print(f"Loaded {len(trained_models)} models from {MODEL_FILENAME}")

test_mat_data = scipy.io.loadmat(TEST_DATA_FILENAME)
TEST_DATA_KEY = 'test_ecog'

test_ecog_data = test_mat_data[TEST_DATA_KEY]
num_subjects = len(test_ecog_data)


predicted_dg_list = []

for subj_idx in range(num_subjects):
    subj_ecog_test = test_ecog_data[subj_idx][0]
    if subj_idx >= len(trained_models) or trained_models[subj_idx] is None:
        target_len = subj_ecog_test.shape[0]
        num_fingers = 5
        final_predictions = np.zeros((target_len, num_fingers))
        predicted_dg_list.append(final_predictions)
        continue

    model = trained_models[subj_idx]

    test_windows = generate_sliding_windows(subj_ecog_test, WINDOW_LEN_MS, OVERLAP_MS, SAMPLING_FREQ)
    test_features = np.array([extract_band_power_features(window, SAMPLING_FREQ) for window in test_windows])

    if len(test_features) >= LAG_STEPS:
        X_test_lagged = np.array([test_features[i:i+LAG_STEPS].flatten() for i in range(len(test_features) - LAG_STEPS + 1)])
    else:
        target_len = subj_ecog_test.shape[0]
        num_fingers = getattr(model, 'n_outputs_', 5)
        final_predictions = np.zeros((target_len, num_fingers))
        predicted_dg_list.append(final_predictions)
        continue 

    raw_predictions = model.predict(X_test_lagged)

    smoothed_predictions = smooth_predictions_moving_average(raw_predictions, SMOOTHING_WINDOW)

    target_len = subj_ecog_test.shape[0]
    total_duration_pred = smoothed_predictions.shape[0] / PREDICTION_FREQ
    _, upsampled_predictions = upsample_predictions_spline(
        smoothed_predictions, PREDICTION_FREQ, SAMPLING_FREQ, total_duration_pred
        )

    final_predictions = pad_array_to_length(upsampled_predictions, target_len)

    predicted_dg_list.append(final_predictions)

cell_array_1xN = np.empty((1, num_subjects), dtype=object)
for i, arr in enumerate(predicted_dg_list):
    cell_array_1xN[0, i] = arr

scipy.io.savemat(
    OUTPUT_FILENAME,
    {'predicted_dg': cell_array_1xN},
    format='5'
)