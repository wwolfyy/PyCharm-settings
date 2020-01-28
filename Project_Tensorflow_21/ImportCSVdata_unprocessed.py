# %% import packages
import os
import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

# %% define functions

def load_data_csv(dir_path, sub_path):
    matfile_path = os.path.join(dir_path, sub_path)
    return pd.read_csv(matfile_path)

def load_data_mat(dir_path, sub_path, string):
    matfile_path = os.path.join(dir_path, sub_path)
    var = spio.loadmat(matfile_path, variable_names=string, chars_as_strings=True)
    return var[string]

def get_data_outlier_clipped(data, iqr_multiplier):
    result = []
    for column in data:
        sorted_data = sorted(data[column])
        q1, q3 = np.percentile(sorted_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)

        result_interim = []
        for y in sorted_data:
            if y < upper_bound and y > lower_bound:
                result_interim.append(y)
            elif y >= upper_bound:
                result_interim.append(upper_bound)
            elif y <= lower_bound:
                result_interim.append(lower_bound)

        result.join(result_interim)
    return result


    data_wo_outlier = get_clipped_iqr_raw(raw_data, lower_bound, upper_bound)
    data_clipped = get_clipped_iqr_data(data_wo_outlier, raw_data, lower_bound, upper_bound)
    return data_clipped

# %% define csv paths for Linux

# DIR_PATH_csv = '/home/lstm/Desktop/GoogleDrive_local/Data Release/CSV' # Linux
# DIR_PATH_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files 8 multiple all dim'
# DIR_PATH_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files 8 multiple 2 dim'

# %% define paths for Windows

DIR_PATH_csv = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV'
# DIR_PATH_mat = r'C:\Users\jp\Google Drive\Wadilt Macro/Quant Project\Outsourcing/Data Release\mat files 8 multiple all dim'
# DIR_PATH_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files 8 multiple 2 dim'
DIR_PATH_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files temp'

# %% define paths for windows on ubunto01

#DIR_PATH_csv = r'C:\Users\jp\Google Drive\Data Release\CSV'
# DIR_PATH_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple all dim'
# DIR_PATH_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple 2 dim'
# DIR_PATH_mat = r'C:\Users\jp\Google Drive\Data Release\mat files temp'

# %% define file path variables

# tp_name = 'KOSPI_PTTP_mrng'
tp_name = 'KOSPI_HLmid_mrng1030'
# tp_name = 'KOSPI_HLmid_mrng'
#tp_name = 'KOSPI_HIGH_mrng1030'

csv_name = tp_name + '.csv'
matfile_name = tp_name + '.mat'

# %% load data
X_df = load_data_csv(DIR_PATH_csv, 'X_unprocessed_' + csv_name)
Y_df = load_data_csv(DIR_PATH_csv, 'Y_unprocessed_' + csv_name)
X_df = X_df.drop('tradedate', axis=1)
Y_df = Y_df.drop('date', axis=1)
X =X_df.values
Y =Y_df.values

MinBatchSize = int(load_data_mat(DIR_PATH_mat, matfile_name, 'MBS'))
#mu = load_data_mat(DIR_PATH_mat, matfile_name, 'mu')[0][0]
#sigma = load_data_mat(DIR_PATH_mat, matfile_name, 'sigma')[0][0]
sequenceSize = load_data_mat(DIR_PATH_mat, matfile_name, 'SequenceSize')[0][0]
train_miniBatchMultiple = load_data_mat(DIR_PATH_mat, matfile_name, 'train_miniBatchMultiple')[0][0]
valid_batchMultiple = load_data_mat(DIR_PATH_mat, matfile_name, 'valid_batchMultiple')[0][0]
batchSize_multiple = load_data_mat(DIR_PATH_mat, matfile_name, 'batchSize_multiple')[0][0]
#tradedates = load_data_mat(DIR_PATH_mat, matfile_name, 'tradedates')

# %% manually change validation and mini batch size if needed

train_miniBatchMultiple = train_miniBatchMultiple
train_miniBatchMultiple = 3
valid_batchMultiple = valid_batchMultiple
#valid_batchMultiple = 4
batchSize_multiple = batchSize_multiple
MinBatchSize = train_miniBatchMultiple * batchSize_multiple

# %% examine distribution

X_df.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.title("Features Original")
plt.show()

Y_df.hist(bins=50)
plt.tight_layout()
plt.title("Target Original")
plt.show()

# %% clip outliers
X = get_data_outlier_clipped(X, 2)
Y = get_data_outlier_clipped(Y, 2)
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)


X = X.apply(get_data_outlier_clipped,2)
Y = Y.apply(get_data_outlier_clipped,2)

print("Feature Data Shape: " + str(X.shape))
print("Traget Data Shape: " + str(Y.shape))

# %% examine distribution again

X_df = DataFrame(X)
Y_df = DataFrame(Y)

X.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.title("Features Clipped")
plt.show()

Y_df.hist(bins=50)
plt.tight_layout()
plt.title("Target Clipped")
plt.show()

# %% standardize

scaler_std_X = StandardScaler().fit(X)
X = scaler_std_X.transform(X)

scaler_std_Y = StandardScaler().fit(Y)
Y = scaler_std_Y.transform(Y)

# %% pad in zeros to make number of features a multiple of 8

# padding = np.zeros([len(X), 6])
# X = np.concatenate((X, padding), axis=1)

# %% divide training, validation and test sets



# # train set index
# dataIDX = 1:size(Y, 1)
# trainIDX = dataIDX(1: end - valid_batchMultiple * batchSize_multiple);
#
# # cut off to multiple of batchSize_multiple
# if ~isequal(mod(length(trainIDX), batchSize_multiple), 0)
#     beginIDX = batchSize_multiple * floor(length(trainIDX) / batchSize_multiple);
#     abandonIDX = trainIDX(1:length(trainIDX) - beginIDX);
#     trainIDX(ismember(trainIDX, abandonIDX)) = [];
#     dataIDX(ismember(dataIDX, abandonIDX)) = [];
# end
#
# % cutoff to include most recent observations
# if isnan(MyTrainOptions.MinBatchSize)
#     MyTrainOptions.MinBatchSize = length(trainIDX);
# end
# if ~isequal(mod(length(trainIDX), MyTrainOptions.MinBatchSize), 0)
#     beginIDX = MyTrainOptions.MinBatchSize * floor(length(trainIDX) / MyTrainOptions.MinBatchSize);
#     abandonIDX = trainIDX(1:length(trainIDX) - beginIDX);
#     trainIDX(ismember(trainIDX, abandonIDX)) = [];
#     dataIDX(ismember(dataIDX, abandonIDX)) = [];
# end
#
# % validation & test set indices
# validIDX = sort(dataIDX(~ismember(dataIDX, trainIDX)), 'ascend');

X =
trainIDX = 576
validIDX =

# assign indices
Y_train = Y[trainIDX]
X_train = X[trainIDX]

Y_valid = Y[validIDX]
X_valid = X[validIDX]


# %% Prepare sequences of data

look_back = sequenceSize  # multiple of 8 for float16 training
num_features = X.shape[1]
num_samples = X.shape[0] - look_back + 1

X_reshaped = np.zeros((num_samples, look_back, num_features))
Y_reshaped = np.zeros((num_samples))

for i in range(num_samples):
    Y_position = i + look_back
    X_reshaped[i] = X[i:Y_position]
    Y_reshaped[i] = Y[Y_position - 1]

print("X_sequence: " + str(X_reshaped.shape))
print("Y_sequence: " + str(Y_reshaped.shape))


FROM HERE-------------------------------------------------
% remove
outlier
dates
if isequal(Asset, 'KOSPI')
if ~exist('outlierDates_KOSPI', 'var')
error('Global variable ''outlierDates_KOSPI'' does not exist. ')
end
outlierDates = outlierDates_KOSPI;
end
if isequal(Asset, 'KGB')
if ~exist('outlierDates_KGB', 'var')
error('Global variable ''outlierDates_KGB'' does not exist.')
end
outlierDates = outlierDates_KGB;
end
OLidx = ismember(Y.(1), outlierDates );
Y = Y(~OLidx,:);
X = X(~OLidx,:);




# %%
# Fit data size multiple of 8 for half-precision training,
# and multiple of mini batch size for stateful LSTM training

cutIDX_begin = 23;

X_reshaped = X_reshaped[cutIDX_begin:]
Y_high_reshaped = Y_high_reshaped[cutIDX_begin:]
Y_low_reshaped = Y_low_reshaped[cutIDX_begin:]

print("X_sequence: " + str(X_reshaped.shape))
print("Y_low_reshaped: " + str(Y_low_reshaped.shape))
print("Y_high_reshaped: " + str(Y_high_reshaped.shape))

# %%
# split = 0.9445 # 90% train, 10% test
# split_idx = int(X_sequence.shape[0]*split)
split_idx_begin = 576

# ...train
X_train_reshaped = X_reshaped[:split_idx_begin]
Y_low_train_reshaped = Y_low_reshaped[:split_idx_begin]
Y_high_train_reshaped = Y_high_reshaped[:split_idx_begin]

# ...test
X_test_reshaped = X_reshaped[split_idx_begin:]
Y_low_test_reshaped = Y_low_reshaped[split_idx_begin:]
Y_high_test_reshaped = Y_high_reshaped[split_idx_begin:]

print("Final Features Train Shape: " + str(X_train_reshaped.shape))
print("Final Targets LOW Train Shape: " + str(Y_low_train_reshaped.shape))
print("Final Targets HIGH Train Shape: " + str(Y_high_train_reshaped.shape))
print("")
print("Final Features Test Shape: " + str(X_test_reshaped.shape))
print("Final Targets LOW Test Shape: " + str(Y_low_test_reshaped.shape))
print("Final Targets HIGH Test Shape: " + str(Y_high_test_reshaped.shape))