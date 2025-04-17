#!/usr/bin/env python
# coding: utf-8

# # BE 521: Final Project Part 1
# Spring 2025
# 
# Adapted by Kevin Xie
# 
# Updated by Zhongchuan Xu
# 
# 32 Points
# 
# Objective: Predict finger movements from ECoG Recordings
# 
# Due: April 10th
# 

# ## Important Deadlines
# * Final Project Part 1 (Canvas)
#  * Due: April 10th
#  * 32 Points
# * Team Registration
#  * Due: April 10th
#  * 5 Points
# * Team Responsibilities (Canvas)
#  * Due: April 14th
#  * 3 Point
# * Checkpoint 1, r > 0.33
#  * Due: April 17th
#  * 20 Points
# * Checkpoint 2, r > 0.45
#  * Due: April 24th
#  * 15 Points
# * End of competition, submit algorithm (Canvas):
#  * Due: April 25th
#  * 15 Points
# * Final Report
#  * Due: April 27th
#  * 60 Points
# * Competition results (Final class session)
#  * On: April 30th
# 
# The grading is structured so that going the extra mile is definitely rewarded. We want you to show what you've learned this semester, and to have some fun!

# ## Writing Your Code
# To get started with the final project we have provided a a series of method stubs for you to fill out. Your job for part 1 of the final project is to build a prediction pipeline that takes in the ECoG and dataglove finger angle recordings (serving as the data and labels respectively), then uses machine learning methods to generate predicted finger angles from the ECoG signals. The functions you will develop in this assignment are as follows:
# * `get_windowed_feats` This function will take in raw ECoG data, and use the 2 following helper functions to filter the data, calculate sliding-window features.
#  * `filter_data` This function will apply a filter to the raw data and return cleaned data
#  * `get_features` This function will take in a window of cleaned data and return a vector of features for that window
# * `create_R_matrix` This function will take in a feature matrix and return a response matrix as an adaptation of the optimal linear decoder method.
# 
# 

# ## Optimal Linear Decoder
# You will use a modified version of the **optimal linear decoder** method as described in Warland et al., 1997. We will recapitulate the method in this section, but consult the paper for more details. Our ultimate goal is to predict the angle of each finger as it moves over time using data recorded from the ECoG channels.
# 
# The position data is captured for 300 seconds, which you will split up into $M$ total time bins, and the number of ECoG channels, $\nu$, is 61, 46, and 64 for subject 1, 2, and 3 respectively.
# 
# The paradigm we adapt here tries to predict finger angle at a given time window using ECoG features calculated over the preceding $N$ time windows, using
# the following steps:
# 
# First, $p$ features will be calculated across all $\nu$ ECoG channels over $M$ total time windows to get a feature matrix of shape $\bigl(M, (\nu \times p)\bigr)$
# 
# Then, following the approach that Warland et al., 1997 takes, we will construct a row vector corresponding to each time bin, that contains features for all the ECoG channels over the preceding *N* time bins (in the paper, spike counts are their features and they index neurons instead of ECoG channels). Thus, there will be a good amount of redundancy between row vectors of adjacent time bins, but that is okay.
# 
# Let $r^{c,\phi}_t$ be the value of the feature in window $t \in \{1,2,\dots,M\}$, channel $c\in\{1,2,\dots,\nu\}$ and with feature $\phi\in\{1,2,\dots,p\}$. Let the response matrix $R \in \mathbb{R}^{M \times (1+N \cdot p \cdot \nu )}$ be defined as:
# 
# $$R = \begin{bmatrix}
# \mathbf{1} & r^{(1,1)}_1 & r^{(1,1)}_1 & \cdots & r^{(1,1)}_1 & r^{(1,1)}_1 & r^{(1,2)}_1 & \cdots & r^{(1,2)}_1 & \cdots & r^{(\nu,p)}_1 & \cdots & r^{(\nu,p)}_1\\
# \mathbf{1} & r^{(1,1)}_1 & r^{(1,1)}_1 & \cdots & r^{(1,1)}_1 & r^{(1,1)}_2 & r^{(1,2)}_1 & \cdots & r^{(1,2)}_2 & \cdots & r^{(\nu,p)}_1 & \cdots & r^{(\nu,p)}_2\\
# \mathbf{1} & r^{(1,1)}_1 & r^{(1,1)}_1 & \cdots & r^{(1,1)}_2 & r^{(1,1)}_3 & r^{(1,2)}_1 & \cdots & r^{(1,2)}_3 & \cdots & r^{(\nu,p)}_1 & \cdots & r^{(\nu,p)}_3\\
# \vdots   & \vdots     & \vdots     & \ddots & \vdots     & \vdots     & \vdots     & \ddots & \vdots   & \cdots & \vdots         & \ddots & \vdots\\
# \mathbf{1} & r^{(1,1)}_1 & r^{(1,1)}_2 & \cdots & r^{(1,1)}_{N-1} & r^{(1,1)}_N & r^{(1,2)}_1 & \cdots & r^{(1,2)}_N & \cdots & r^{(\nu,p)}_1 & \cdots & r^{(\nu,p)}_N\\
# \mathbf{1} & r^{(1,1)}_2 & r^{(1,1)}_3 & \cdots &r^{(1,1)}_{N} & r^{(1,1)}_{N+1} & r^{(1,2)}_2 & \cdots & r^{(1,2)}_{N+1} & \cdots & r^{(\nu,p)}_2 & \cdots & r^{(\nu,p)}_{N+1}\\
# \vdots   & \vdots     & \vdots     & \ddots & \vdots     & \vdots     & \vdots     & \ddots & \vdots   & \cdots & \vdots         & \ddots & \vdots\\
# \mathbf{1} & r^{(1,1)}_{M-N+1} & r^{(1,1)}_{M-N+2} & \cdots & r^{(1,1)}_{M-1} & r^{(1,1)}_M & r^{(1,2)}_{M-N+1} & \cdots & r^{(1,2)}_M & \cdots & r^{(\nu,p)}_{M-N+1} & \cdots & r^{(\nu,p)}_M\\
# \end{bmatrix}$$
# 
# 

# This is also referred to as the design or feature matrix, with each column being a predictor, or feature. The column of 1’s accounts for the intercept term in linear regression/decoding. Here we are repeating the first windows $N-1$ times as the padding of the first windows. Make sure you understand what this matrix means before moving on.
# 
# We denote the target matrix as $Y \in \mathbb{R}^{M \times 5}$ and the prediction matrix (e.g. the predicted finger angles) as $\hat{Y} \in \mathbb{R}^{M \times 5}$. Note that in Warland et al., 1997, this quantity is referred to as the stimulus vector since they are talking about decoding the stimulus from neural data after it. We, on the other hand, are trying to decode finger positions using the ECoG data before it, but we can conveniently use the same method. Our goal is to find some optimal weight matrix or filter $f \in \mathbb{R}^{(1+N \cdot p \cdot \nu ) \times 5}$ that minimizes the mean squared error:
# 
# $$f^* = \operatorname{argmin}_{f} \, \mathcal{L}(f) = \operatorname{argmin}_{f} \left\|Y - \hat{Y}\right\|^2,$$
# where $\hat{Y} = Rf$
# 
# We start with the mean squared error (MSE) loss function:
# $$
# \mathcal{L}(f) = \left\|Y - R\,f\right\|^2 = (Y - R\,f)^\top (Y - R\,f)
# $$
# 
# To minimize the loss, we take the derivative with respect to the weight matrix $f$ and set it equal to zero:
# $$
# \frac{\partial \mathcal{L}}{\partial f} = -2\,R^\top (Y - R\,f) = 0
# $$
# 
# This implies:
# $$
# R^\top Y = R^\top R\,f
# $$
# 
# Assuming that $R^\top R$ is invertible, we solve for $f$:
# $$
# f = \left(R^\top R\right)^{-1} R^\top Y
# $$
# 
# This is the the analytic form for the optimal filter $f$ that minimizes the MSE loss.
# 
# This equation should take a familiar form. Warland et al., 1997 don’t refer to it as such, but this is exactly the same as linear regression, one of the most commonly used algorithms in practical machine learning. Not only is this algorithm remarkably powerful, but it has a beautiful analytic form for learning the “weights” (here, the $f$ matrix), a rarity in a field where almost all optimizations involve some sort of iterative algorithm. After learning the filter weights $f$, we can calculate the optimal predictions as: $$\hat{Y} = Rf$$

# ## Dataset
# The dataset for part 1 is stored within `final_proj_part1_data.pkl`. The `.pkl` file type is a pickle file, which stores python objects. You can open the `.pkl` file with this code.
# ```
# with open('final_proj_part1_data.pkl', 'rb') as f:
#   proj_data = pickle.load(f)
# ```
# This stores the data inside the file as a variable named proj_data.
# 
# **NOTE: Python versions don't pickle with each other very well. This pickle file was made in Google Colab. If you are running your own installation of Python and cannot load the file, we recommend you use Colab**
# 
# There are 3 subjects, each with their own Data Glove data (the glove they used to capture hand movements), and ECoG data. The data is represented as a dictionary with keys `'data_glove'` and `'ecog'`, storing the data glove and ecog data, respectively. These keys map to python lists of 3 items. Each item is an np.ndarray corresponding to a subject's data. See the pseudcode below.
# 
# ```
# proj_data = {
#   'data_glove':[np.ndarray for subject 1, np.ndarray for subject 2, np.ndarray for subject 3],
#   'ecog':[np.ndarray for subject 1, np.ndarray for subject 2, np.ndarray for subject 3]
# }
# ```
# 
# All np.ndarray shapes for `'data_glove'` should be $(T,5)$, where $T$ is the number of samples in the signal, and 5 is the number of fingers.
# 
# The np.ndarray shapes for `'ecog'` are $(T, 61)$, $(T, 46)$, and $(T,64)$, where T is the number of samples in the signal, and each subject had 61, 46, and 64 ecog channels, respectively.
# 
# **The sampling rate of the data glove and ecog was 1000 Hz**
# 
# <!-- The dataset is also on IEEG
# * Subject 1
#  * I521_Sub1_Training_ecog - Training ECoG \
#  * I521_Sub1_Training_dg - Training Data Glove \
#  * I521_Sub1_Leaderboard_ecog - Testing ECoG
# * Subject 2
#  * I521_Sub2_Training_ecog - Training ECoG \
#  * I521_Sub2_Training_dg - Training Data Glove \
#  * I521_Sub2_Leaderboard_ecog - Testing ECoG
# * Subject 3
#  * I521_Sub3_Training_ecog - Training ECoG \
#  * I521_Sub3_Training_dg - Training Data Glove \
#  * I521_Sub3_Leaderboard_ecog - Testing ECoG -->
# 
# Your task is to develop an algorithm to use the ECoG to predict finger movements that are captured by the Data Glove.

# In[1]:


import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
from scipy import signal as sig


# In[31]:


train_file = 'raw_training_data.mat' # file path
test_file = 'leaderboard_data.mat'

# load training data
train_data = sio.loadmat(train_file) # returns dict with keys and values as numpy arrays
ecog = train_data['train_ecog']
data_glove = train_data['train_dg']

# leaderboard testing data
test_data = sio.loadmat(test_file)
leaderboard_ecog = test_data['leaderboard_ecog']


# In[49]:


train_ecog = []
test_ecog = []
train_glove = []
test_glove = []

train_len = int(0.8*len(ecog[0][0]))

for subject_idx in range(3):
    ecog_data = ecog[subject_idx]
    glove_data = data_glove[subject_idx]
    train_ecog.append(ecog_data[0][:train_len])
    test_ecog.append(ecog_data[0][train_len:])
    train_glove.append(glove_data[0][:train_len])
    test_glove.append(glove_data[0][train_len:])


# In[54]:


def filter_data(raw_eeg, fs=1000):
  """
  Write a filter function to clean underlying data.
  Filter type and parameters are up to you. Points will be awarded for reasonable filter type, parameters and application.
  Please note there are many acceptable answers, but make sure you aren't throwing out crucial data or adversly
  distorting the underlying data!

  Input:
    raw_eeg (samples x channels): the raw signal
    fs: the sampling rate (1000 for this dataset)
  Output:
    clean_data (samples x channels): the filtered signal
  """
  def notch_filter(data, freq, fs, Q=30):
      """Apply a notch filter at a specific frequency."""
      b, a = sig.iirnotch(w0=freq/(fs/2), Q=Q)
      return sig.filtfilt(b, a, data, axis=0)

  def bandpass_filter(data, lowcut, highcut, fs, order=4):
      """Apply a Butterworth bandpass filter."""
      nyq = 0.5 * fs
      low = lowcut / nyq
      high = highcut / nyq
      b, a = sig.butter(order, [low, high], btype='band')
      return sig.filtfilt(b, a, data, axis=0)

    # Apply notch filters at 60 Hz harmonics (up to 300 Hz)
  filtered = raw_eeg.copy()
  for freq in [60, 120, 180, 240, 300]:
      filtered = notch_filter(filtered, freq, fs)

    # Apply bandpass filter (default 1–200 Hz)
  clean_data = bandpass_filter(filtered, lowcut=1, highcut=200, fs=fs)

  return clean_data


# In[61]:


#Your code here
train_ecog_0 = train_ecog[0][0]


# In[72]:


train_ecog[2][0]


# In[59]:


filtered_0 = filter_data(train_ecog[0][0])


# In[73]:


train_ecog_0


# In[74]:


plt.figure(figsize=(15, 6))
plt.plot(train_ecog_0, label='Raw Channel 0', alpha=0.5)
plt.plot(filtered_0, label='Filtered Channel 0', linewidth=1)
plt.title('Comparison of Raw vs Filtered ECoG Signal (Channel 0)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()


# I used a filter frequency of 1-200 Hz for the bandpass filter and a notch filter at 60Hz harmonics up to 300 Hz.

# # 2. Calculating Features (12 points)
# 
# Here you will complete the `get_windowed_feats` and `get_features` functions.

# ## 1.
# We will calculate features across sliding time windows. if we use a suggested window length of 100ms with a 50ms window overlap, how many feature windows, $M$, will we have if we computed features using all the data in a given subject? Feel free to re-use code from previous homeworks.

# In[78]:


fs = 1000
xLen = len(train_ecog[0])


# In[79]:


#Your code here
winLen= 0.1
winDisp = 0.05

def NumWins(xLen, fs, winLen, winDisp):
  winLen = winLen * fs
  winDisp = winDisp * fs
  return int((xLen - winLen) // winDisp + 1)


# In[80]:


NumWins(xLen, fs, winLen, winDisp)


# In[81]:


def moving_win_feats(x, fs, winLen, winDisp, featFn):
  numWins = NumWins(len(x), fs, winLen, winDisp)
  winLen = winLen * fs
  winDisp = winDisp * fs
  feature_values = np.zeros(int(numWins))
  for i in range(int(numWins)):
    start = int(i * winDisp)
    end = int(start + winLen)
    feature_values[i] = featFn(x[start:end])
  return feature_values


# In[84]:


from scipy.stats import skew, kurtosis
def bandpower(data, fs, fmin, fmax):
      f, Pxx = sig.welch(data, fs=fs, nperseg=min(256, len(data)), nfft=1024)
      band = (f >= fmin) & (f <= fmax)
      return np.trapz(Pxx[band], f[band])

def get_features(filtered_window, fs=1000):
  """
    Write a function that calculates features for a given filtered window.
    Feel free to use features you have seen before in this class, features that
    have been used in the literature, or design your own!

    Input:
      filtered_window (window_samples x channels): the window of the filtered ecog signal
      fs: sampling rate
    Output:
      features (channels x num_features): the features calculated on each channel for the window
  """
  
  window_samples, num_channels = filtered_window.shape
  num_features = 6
  features = np.zeros((num_channels, num_features))

  for ch in range(num_channels):
      signal = filtered_window[:, ch]
      features[ch, 0] = np.mean(signal)                        # average
      # commented out a couple of features because I wanted to keep it the same as the paper for now but will uncomment out for the future trials
      # features[ch, 1] = np.std(signal)                         # STD
      # features[ch, 2] = skew(signal)                           # Skewness
      # features[ch, 3] = kurtosis(signal)                       # Kurtosis
      features[ch, 1] = bandpower(signal, fs, 5, 15)           # Mu band
      features[ch, 2] = bandpower(signal, fs, 20, 25)          # Beta band
      features[ch, 3] = bandpower(signal, fs, 75, 115)         # High gamma
      features[ch, 4] = bandpower(signal, fs, 125, 160)        # Upper high gamma
      features[ch, 5] = bandpower(signal, fs, 160, 175)
  return features


# In[85]:


def get_windowed_feats(raw_ecog, fs, window_length, window_overlap):
  """
    Write a function which processes data through the steps of filtering and
    feature calculation and returns features. Points will be awarded for completing
    each step appropriately (note that if one of the functions you call within this script
    returns a bad output, you won't be double penalized). Note that you will need
    to run the filter_data and get_features functions within this function.

    Inputs:
      raw_eeg (samples x channels): the raw signal
      fs: the sampling rate (1000 for this dataset)
      window_length: in seconds
      window_overlap: in seconds
    Output:
      all_feats (num_windows x (channels x features)): the features for each channel for each time window
        note that this is a 2D array.
  """
  filtered = filter_data(raw_ecog, fs)

  window_length = int(window_length * fs)
  window_overlap = int(window_overlap * fs)

  step = window_length - window_overlap
  num_samples, _ = filtered.shape
  starts = np.arange(0, num_samples - window_length + 1, step)

  all_feats = []

  for start in starts:
      end = start + window_length
      window = filtered[start:end, :]
      feats = get_features(window, fs)
      all_feats.append(feats.flatten())

  return np.array(all_feats)


# In[86]:


#Your code here
Num_columns = 62 * 6 * 3 + 1
print(Num_columns)


# The size of our matrix would be 5999 x 1117. 

# In[15]:


def create_R_matrix(features, N_wind):
  """
  Write a function to calculate the R matrix

  Input:
    features (num_windows x (channels x features)):
      the features you calculated using get_windowed_feats
    N_wind: number of windows to use in the R matrix

  Output:
    R (samples x (N_wind*channels*features))
  """
  M, D = features.shape

  padding = np.repeat(features[[0]], repeats=N_wind - 1, axis=0)  # shape: (N-1, D)
  padded_feats = np.vstack((padding, features))  # shape: (M + N - 1, D)

  R = np.zeros((M, N_wind * D))

  for i in range(M):
      window = padded_feats[i:i + N_wind, :]
      R[i, :] = window.flatten(order='C')
  bias = np.ones((M, 1))
  R = np.hstack((bias, R))
  return R


# In[88]:


train_ecog[0]


# In[89]:


train_ecog = train_ecog


# In[90]:


test_ecog = test_ecog


# In[91]:


#Your code here
def compute_linear_filter(R, finger_flexion_raw, original_fs=1000, window_step=50):

    M = R.shape[0]
    T = finger_flexion_raw.shape[0]
    Y = sig.resample(finger_flexion_raw, len(R))

    alpha = 0
    RTR = R.T @ R + alpha * np.eye(R.shape[1])
    return np.linalg.inv(RTR) @ (R.T @ Y)


# In[92]:


train_feats_1 = [get_windowed_feats(train_ecog_thing, fs=1000, window_length=0.1, window_overlap=0.05) for train_ecog_thing in train_ecog]
test_feats_2 = [get_windowed_feats(test_ecog_thing, fs=1000, window_length=0.1, window_overlap=0.05) for test_ecog_thing in test_ecog]
    


# In[123]:


# test ecog r matrix
R_train = [create_R_matrix(train_feats_thing, N_wind=3) for train_feats_thing in train_feats_1]
R_test = [create_R_matrix(test_feats_thing, N_wind=3) for test_feats_thing in test_feats_2]


# In[124]:


R_train[0]


# In[125]:


f = [compute_linear_filter(R_train_thing, train_glove_thing) for R_train_thing, train_glove_thing in zip(R_train, train_glove)]


# ## 2.
# Try one other machine learning models using your features and finger angle labels. Look back through previous homeworks to get some ideas

# Ridge Regression

# In[126]:


#your code here
from sklearn.linear_model import Ridge

def train_ridge_model(R, Y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(R, Y)
    return model


# Random Forest Regressor

# In[127]:


from sklearn.ensemble import RandomForestRegressor

def train_rf_model(R, Y):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(R, Y)
    return model


# Multi-layer perceptron

# In[128]:


from sklearn.neural_network import MLPRegressor

def train_mlp_model(R, Y):
    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='adam', max_iter=500, random_state=42)
    model.fit(R, Y)
    return model


# ## 3.
# Produce predictions on the testing set for each finger angle.
# 
# Report your correlations here using the linear filter, and when using the other model(s) that you tried, as follows:
# 
# > For each subject, calculate the correlation coefficient between the predicted and test finger angles for each finger separately.
# 
# > You therefore should have 15 correlations: five per subject, with three subjects.
# 
# You will find  [pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) to be helpful and already imported.

# In[129]:


R_test[0]


# In[130]:


f[0]


# In[134]:


R_test[0] @ f[0]


# In[132]:


test_glove[0]


# In[133]:


fingers = np.array([0, 1, 2, 4])


# In[141]:


#Your code here
# this is the first part for the method described in the paper
correlations = []

for subject_idx in range(3):
    R_test_subject = R_test[subject_idx]
    test_glove_subject = test_glove[subject_idx]

    downsampled_glove = sig.resample(test_glove_subject, len(R_test_subject))
    print(len(downsampled_glove))
    Y_pred =  R_test_subject @ f[subject_idx]
    
    finger_correlations = []
    for finger_idx in range(5):
        r, _ = pearsonr(Y_pred[:, finger_idx], downsampled_glove[:, finger_idx])
        finger_correlations.append(r)
    
    correlations.append(finger_correlations)
    print(f"\nSubject {subject_idx + 1} finger correlations:")
    for finger, r in enumerate(finger_correlations):
        print(f"Finger {finger + 1}: {r:.3f}")

correlations_np = np.array(correlations)
avg_wout_4th_finger = np.mean(correlations_np[:, fingers])
avg_correlation = np.mean(correlations_np)
print(f"\nAverage correlation across all subjects and fingers: {avg_correlation:.3f}")
print(f"\nAverage correlation without 4th finger: {avg_wout_4th_finger:.3f}")


# Using the linear filter model described in the paper, the average correlation that I got was 0.356 (I did a training, test split of 80, 20). This is not the worst, but I will try to do a bit better using the other models that I had above. 

# In[142]:


R_train[0]


# In[105]:


train_glove[0]


# In[148]:


leaderboard_feats = [get_windowed_feats(leaderboard_thing[0], fs=1000, window_length=0.1, window_overlap=0.05) for leaderboard_thing in leaderboard_ecog]


# In[149]:


R_leaderboard = [create_R_matrix(test_feats_thing, N_wind=3) for test_feats_thing in leaderboard_feats]


# In[150]:


linear_correlations = []
ridge_correlations = []
rf_correlations = []
mlp_correlations = []

for subject_idx in range(3):
    print(f"\nSubject {subject_idx + 1}:")
    
    R_train_subject = R_train[subject_idx]
    R_test_subject = R_test[subject_idx]
    train_glove_subject = train_glove[subject_idx]
    test_glove_subject = test_glove[subject_idx]

    train_glove_downsampled = sig.resample(train_glove_subject, len(R_train_subject))
    test_glove_downsampled = sig.resample(test_glove_subject, len(R_test_subject))

    # Linear Filter
    f = compute_linear_filter(R_train_subject, train_glove_downsampled)
    Y_pred_linear = R_test_subject @ f
    
    # Ridge Regression
    ridge_model = train_ridge_model(R_train_subject, train_glove_downsampled)
    Y_pred_ridge = ridge_model.predict(R_test_subject)
    
    # Random Forest
    rf_model = train_rf_model(R_train_subject, train_glove_downsampled)
    Y_pred_rf = rf_model.predict(R_test_subject)
    
    # MLP
    mlp_model = train_mlp_model(R_train_subject, train_glove_downsampled)
    Y_pred_mlp = mlp_model.predict(R_test_subject)

    # correlations for each model and fiinds
    for model_name, Y_pred in [
        ('Linear Filter', Y_pred_linear),
        ('Ridge', Y_pred_ridge),
        ('Random Forest', Y_pred_rf),
        ('MLP', Y_pred_mlp)
    ]:
        print(f"\n{model_name}:")
        finger_correlations = []
        for finger_idx in range(5):
            r, _ = pearsonr(Y_pred[:, finger_idx], test_glove_downsampled[:, finger_idx])
            finger_correlations.append(r)
            print(f"Finger {finger_idx + 1}: {r:.3f}")
        correlations_np = np.array(finger_correlations)
        avg_wout_4th_finger = np.mean(correlations_np[fingers])
        avg_correlation = np.mean(correlations_np)
        print(f"\nAverage correlation across all subjects and fingers: {avg_correlation:.3f}")
        print(f"\nAverage correlation without 4th finger: {avg_wout_4th_finger:.3f}")


# In[154]:


all_predictions = []

for subject_idx in range(3):
    subject_data = R_leaderboard[subject_idx]
    subject_predictions = rf_model.predict(subject_data)
    all_predictions.append(subject_predictions)

predicted_dg = np.array(all_predictions, dtype=object)
sio.savemat('predictions.mat', {'predicted_dg': predicted_dg})

