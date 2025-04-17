import pickle
import scipy.io as sio
from scipy.io import savemat

# Load the trained predictor
with open('algorithm.pkl', 'rb') as f:
    predictor = pickle.load(f)

# Load leaderboard data
test_file = 'leaderboard_data.mat'
test_data = sio.loadmat(test_file)
leaderboard_ecog = test_data['leaderboard_ecog']

# Make predictions
predicted_dg = predictor.predict(leaderboard_ecog)

# Save predictions
savemat('predictions.mat', {'predicted_dg': predicted_dg})