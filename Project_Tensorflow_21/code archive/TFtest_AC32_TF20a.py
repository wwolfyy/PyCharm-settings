# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:12:40 2019

@author: jp
"""

#%%
# Prepare Tensorflow and test if the module is working    

import tensorflow as tf

tf.config.gpu.set_per_process_memory_fraction(0.5)
tf.config.gpu.set_per_process_memory_growth(True)

print(tf.config.gpu.get_per_process_memory_fraction())
print(tf.config.gpu.get_per_process_memory_growth())

print(tf.__version__)
print(tf.keras.__version__)

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

print(hello_constant)

import os

#%%
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()
tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)

#%%
#from tensorflow.keras import backend as K
#dtype='float16'
#K.set_floatx(dtype)
## default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
#K.set_epsilon(1e-4) 

#%%
# Provide method to load data from the respective csv files

import pandas as pd

DIR_PATH = '/home/lstm/Desktop/GoogleDrive_local/JP data on ubuntu01'

def load_data(dir_path=DIR_PATH):
    csv_path = os.path.join(dir_path, 'X_unprocessed_KOSPI_HIGH_mrng1030.csv')
    return pd.read_csv(csv_path)

def load_low_data(dir_path=DIR_PATH):
    csv_path = os.path.join(dir_path, 'Y_unprocessed_KOSPI_LOW_mrng1030.csv')
    return pd.read_csv(csv_path)

def load_high_data(dir_path=DIR_PATH):
    csv_path = os.path.join(dir_path, 'Y_unprocessed_KOSPI_HIGH_mrng1030.csv')
    return pd.read_csv(csv_path)


#%%
# Load data and take a peek
stock = load_data()
low = load_low_data()
high = load_high_data()
#stock.head()
#low.plot()
#high.plot()


#%%
# Drop the tradedate column, because it won't be used
stock_df = stock.drop('tradedate', axis=1)
low_df = low.drop('date', axis=1)
high_df = high.drop('date', axis=1)

#%%
# Show the distribution both from train and test data
from pandas import DataFrame
import matplotlib.pyplot as plt

X_stock_df = DataFrame(stock_df)
#X_stock_df.hist(bins=50, figsize=(20, 15))
#plt.tight_layout()

low_df.hist(bins=50)
plt.tight_layout()
plt.title("Low Data Histogram - Raw Original")
plt.show()

high_df.hist(bins=50)
plt.tight_layout()
plt.title("High Data Histogram - Raw Original")
plt.show()

#%%
# Only for testing
#plt.hist(X_train_df[0], bins=50)
#plt.title("Testing Histogram - Raw Original")
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.show()


#%%
# Custom method
# Used to cut the outliers and 
# replace them with min/max value from each feature data column independently

import numpy as np
#from statistics import mean 

def detect_outlier_iqr(data, lower_bound, upper_bound):
    outliers=[]

    for y in data:
        if y >= upper_bound or y <= lower_bound:
            outliers.append(y)
    return outliers

# Give list without outlier
def get_clipped_iqr_raw(data, lower_bound, upper_bound):
    result=[]

    for y in data:
        if y < upper_bound and y > lower_bound:
            result.append(y)
    return result

# Give list with outlier replaced with mean value
def get_clipped_iqr_data(data, raw, lower_bound, upper_bound):
    result=[]

    min_val = min(data)
    max_val = max(data)
    print("Lowest: " + str(min_val))
    print("Highest: " + str(max_val))

    for y in raw:
        if y < upper_bound and y > lower_bound:
            result.append(y)
        elif y >= upper_bound:
            result.append(max_val)
        elif y <= lower_bound:
            result.append(min_val)
    
    return result

def get_data_outlier_clipped(raw_data):
    print("")
    print("Data Index: " + str(raw_data.name))
    sorted_raw_data = sorted(raw_data)
    
    q1, q3= np.percentile(sorted_raw_data,[25,75])
    print("Q1: " + str(q1))
    print("Q3: " + str(q3))
    iqr = q3 - q1
    print("IQR: " + str(iqr))
    
    # Bigger multiplier means wider range
    iqr_multiplier = 2
    lower_bound = q1 - (iqr_multiplier * iqr) 
    upper_bound = q3 + (iqr_multiplier * iqr) 
    print("Lower Bound: " + str(lower_bound))
    print("Upper Bound: " + str(upper_bound))
    
    outlier_datapoints = detect_outlier_iqr(raw_data, lower_bound, upper_bound)
    print("Outlier Found(s): " + str(len(outlier_datapoints)))
    print("Outlier(s): ")
    print(np.asarray(outlier_datapoints))
    
    # Data without outlier
    data_wo_outlier = get_clipped_iqr_raw(raw_data, lower_bound, upper_bound)
    # Data with outlier replaced with mean value
    data_clipped = get_clipped_iqr_data(data_wo_outlier, raw_data, lower_bound, upper_bound)
    print("Data Length: " + str(len(data_clipped)))
    return data_clipped


#%%
# Only for testing
#data_clipped = get_data_outlier_clipped(X_train_df[0])
#plt.hist(data_clipped, bins=50)
#plt.title("Testing Histogram - Clipped")
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.show()


#%%
# Get clipped data from train data

X_stock_clipped_df = X_stock_df.apply(get_data_outlier_clipped)
Y_low_clipped_df = low_df.apply(get_data_outlier_clipped)
Y_high_clipped_df = high_df.apply(get_data_outlier_clipped)

# Note
# The first row somehow executed twice, here is the reason:
# In the current implementation apply calls func twice on the first column/row to decide
# whether it can take a fast or slow code path. 
# This can lead to unexpected behavior if func has side-effects,
# as they will take effect twice for the first column/row.
# But it won't affect the final dataframe, at least for this case


#%%
# Show distribution graph of clipped data from train data

#X_stock_clipped_df.hist(bins=50, figsize=(20, 15))
#plt.tight_layout()
#print("Data Histogram - Clipped")
#plt.show()
#
#Y_low_clipped_df.hist(bins=50)
#plt.tight_layout()
#plt.title("Low Data Histogram - Clipped")
#plt.show()
#
#Y_high_clipped_df.hist(bins=50)
#plt.tight_layout()
#plt.title("High Data Histogram - Clipped")
#plt.show()

#%%
# Standardize all data
# Test data will use scaler from it's respective train data

from sklearn.preprocessing import StandardScaler

# Standard scaler with zero mean and 1 as std 
scaler_std_X = StandardScaler().fit(X_stock_clipped_df)
X_stock_scaled_clipped = scaler_std_X.transform(X_stock_clipped_df)

low_scaler_std_X = StandardScaler().fit(Y_low_clipped_df)
Y_low_scaled_clipped = low_scaler_std_X.transform(Y_low_clipped_df)

high_scaler_std_X = StandardScaler().fit(Y_high_clipped_df)
Y_high_scaled_clipped = high_scaler_std_X.transform(Y_high_clipped_df)

print("")
print("All Data has been Clipped and Standardized")
print("")


#%%
# Print shape to see if the qty is correct 

print("Feature Data Shape: " + str(X_stock_scaled_clipped.shape))
print("Low Data Shape: " + str(Y_low_scaled_clipped.shape))
print("High Data Shape: " + str(Y_high_scaled_clipped.shape))


#%%
# Prepare sequences of data
# which look back up to N data to the past

look_back = 100
num_features = X_stock_scaled_clipped.shape[1]
num_samples = X_stock_scaled_clipped.shape[0] - look_back + 1

X_reshaped = np.zeros((num_samples, look_back, num_features))
Y_low_reshaped = np.zeros((num_samples))
Y_high_reshaped = np.zeros((num_samples))

for i in range(num_samples):
    Y_position = i + look_back
    X_reshaped[i] = X_stock_scaled_clipped[i:Y_position]
    Y_low_reshaped[i] = Y_low_scaled_clipped[Y_position-1]
    Y_high_reshaped[i] = Y_high_scaled_clipped[Y_position-1]

print("X_sequence: " + str(X_reshaped.shape))
print("Y_low_reshaped: " + str(Y_low_reshaped.shape))
print("Y_high_reshaped: " + str(Y_high_reshaped.shape))

#%%
# Fit data size multiple of 8 for half-precision training, 
# and multiple of mini batch size for stateful LSTM training

#X_sequence = X_sequence[3:604]
#Y_high_reshaped = Y_high_reshaped[3:604]
#Y_low_reshaped = Y_low_reshaped[3:604]

X_reshaped = X_reshaped[0:]
Y_high_reshaped = Y_high_reshaped[0:]
Y_low_reshaped = Y_low_reshaped[0:]

print("X_sequence: " + str(X_reshaped.shape))
print("Y_low_reshaped: " + str(Y_low_reshaped.shape))
print("Y_high_reshaped: " + str(Y_high_reshaped.shape))

#%%
#split = 0.9445 # 90% train, 10% test
#split_idx = int(X_sequence.shape[0]*split)
split_idx = 576

# ...train
X_train_reshaped = X_reshaped[:split_idx]
Y_low_train_reshaped = Y_low_reshaped[:split_idx]
Y_high_train_reshaped = Y_high_reshaped[:split_idx]

# ...test
X_test_reshaped = X_reshaped[split_idx:]
Y_low_test_reshaped = Y_low_reshaped[split_idx:]
Y_high_test_reshaped = Y_high_reshaped[split_idx:]

print("Final Features Train Shape: " + str(X_train_reshaped.shape))
print("Final Targets LOW Train Shape: " + str(Y_low_train_reshaped.shape))
print("Final Targets HIGH Train Shape: " + str(Y_high_train_reshaped.shape))
print("")
print("Final Features Test Shape: " + str(X_test_reshaped.shape))
print("Final Targets LOW Test Shape: " + str(Y_low_test_reshaped.shape))
print("Final Targets HIGH Test Shape: " + str(Y_high_test_reshaped.shape))


#%%
# Custome method to get metrics for R-squared
def r_square(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


#%%
# import layers & create model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
#tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
from time import time
import datetime
log_dir_ = os.path.join( "TensorFlow","graph",datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir_)

# from tensorflow.keras import models
# from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
#from tensorflow.keras import callbacks

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


#%%
# define parameters
#early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
num_epoch = 10
#num_hidden_neuron = 1024
output_unit = 1
droprate = 0.8

#%%	
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
#tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
##tf.keras.mixed_precision.experimental.set_policy('infer')
#tf.keras.mixed_precision.experimental.global_policy()

#%%
#tf.cast(X_train_reshaped, tf.float16)
#tf.cast(Y_high_train_reshaped, tf.float16)
#tf.cast(X_test_reshaped, tf.float16)
#tf.cast(Y_high_test_reshaped, tf.float16)

#tf.cast(X_train_reshaped, tf.float32)
#tf.cast(Y_high_train_reshaped, tf.float32)
#tf.cast(X_test_reshaped, tf.float32)
#tf.cast(Y_high_test_reshaped, tf.float32)

#%%
#construct layers and compile
model = Sequential()
model.add(LSTM(720, return_sequences = True,
                    stateful = True, batch_input_shape= (32,100,122),
                    kernel_regularizer = regularizers.l2(0.0001),
                    recurrent_regularizer = regularizers.l2(0.0001)))
##model.add(BatchNormalization())
##model.add(ELU())
model.add(Dropout(droprate))

model.add(LSTM(480, return_sequences = False,
                    stateful = True, batch_input_shape= (32,100,122),
                    kernel_regularizer = regularizers.l2(0.0001),
                    recurrent_regularizer = regularizers.l2(0.0001)))
##model.add(BatchNormalization())
##model.add(ELU())
model.add(Dropout(droprate))

model.add(Dense(units=122, kernel_regularizer = regularizers.l2(0.0001)))
model.add(BatchNormalization())
#model.add(ELU())
model.add(Dropout(droprate))

model.add(Dense(units=output_unit, kernel_regularizer = regularizers.l2(0.0001)))
model.add(Activation("linear"))
model.compile(optimizer = adam, loss = 'mean_squared_error', metrics=[r_square, 'mae'])

#%%
model.summary()

#%%
history = model.fit(X_train_reshaped, Y_high_train_reshaped, 
                    validation_data = (X_test_reshaped, Y_high_test_reshaped),
                    batch_size = 32, epochs = num_epoch, shuffle = False, callbacks = [tbCallBack])

#%% ways to clear gpu memory

del model

import gc
gc.collect()

from keras import backend as K
K.clear_session()

#%%
# Training Model for HIGH
# history = model.fit(X_train_reshaped, Y_high_train_reshaped, 
#                     validation_data = (X_test_reshaped, Y_high_test_reshaped),
#                     batch_size = 32, epochs = num_epoch)
#tensorboard --logdir ./Graph


#%%
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss - HIGH')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

#%%
# summarize history for r-square
plt.plot(history.history['r_square'])
plt.plot(history.history['val_r_square'])
plt.title('Model R^2 on Training - HIGH')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


#%%
# See metrics from the prediction result 
high_predictions = model.predict(X_train_reshaped)
lin_mse = mean_squared_error(Y_high_train_reshaped, high_predictions)
lin_rmse = np.sqrt(lin_mse)
print("High Target RMSE: " + str(lin_rmse))

lin_mae = mean_absolute_error(Y_high_train_reshaped, high_predictions)
print("High Target MAE: " + str(lin_mae))

R2 = r2_score(Y_high_train_reshaped, high_predictions)
print("High Target R-Squared: " + str(R2))


#%%
# Make the first plot 
plt.subplot(2, 1, 1)
plt.plot(Y_high_train_reshaped) 
plt.title('All High Target Actual')  
   
# Make the second plot. 
plt.subplot(2, 1, 2) 
plt.plot(high_predictions) 
plt.title('All High Target Prediction')  
   
# Show the figure. 
plt.tight_layout()
plt.show()


#%%
# Do some little test, and compare between prediction with real data
some_labels = Y_high_train_reshaped[:10]
some_data = X_train_reshaped[:10]
some_predict = model.predict(some_data)

print("High Target Predictions:\t\n", some_predict)
print("High Target Labels:\t\n", np.array(some_labels))

plt.plot(some_labels) 
plt.plot(some_predict) 
plt.title('Test Target High')    
   
plt.legend(['Actual', 'Prediction'], loc='upper left')

# Show the figure. 
plt.show()


#%%
# Test Part

high_test_predictions = model.predict(X_test_reshaped)
lin_mse = mean_squared_error(Y_high_test_reshaped, high_test_predictions)
lin_rmse = np.sqrt(lin_mse)
print("High Target RMSE on Test Data: " + str(lin_rmse))

lin_mae = mean_absolute_error(Y_high_test_reshaped, high_test_predictions)
print("High Target MAE on Test Data: " + str(lin_mae))

R2 = r2_score(Y_high_test_reshaped, high_test_predictions)
print("High Target R-Squared on Test Data: " + str(R2))

plt.plot(Y_high_test_reshaped) 
plt.plot(high_test_predictions) 
plt.title('Test Target High on Test Data')    
   
plt.legend(['Actual', 'Prediction'], loc='upper left')

# Show the figure. 
plt.show()


