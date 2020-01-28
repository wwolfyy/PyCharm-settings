def ImportCSVdata_processed(path_CSV, path_mat, tpName, varname, MinBatchMultiple=0, ValidBatchMultiple=0, TestSize=0):
    # inputs
        # tpName: e.g. 'KOSPI_HLmid_mrng1030'
            # 'KOSPI_PTTP_mrng'
            # 'KOSPI_HLmid_mrng'
            # 'KOSPI_HIGH_mrng1030'
        # platform: e.g. Windows8700k, Ubuntu01, WindowsOnUnbuntu01

    # import packages
    import os
    import numpy as np
    import pandas as pd
    import scipy.io as spio
    import math
    import matplotlib.pyplot as plt
    from pandas import DataFrame

    # define functions
    def load_data_csv(dir_path, sub_path):
        matfile_path = os.path.join(dir_path, sub_path)
        return pd.read_csv(matfile_path)

    def load_data_mat(dir_path, sub_path, string):
        matfile_path = os.path.join(dir_path, sub_path)
        var = spio.loadmat(matfile_path, variable_names=string, chars_as_strings=True)
        return var[string]

    # define file path variables
    DIR_PATH_csv = path_CSV
    DIR_PATH_mat = path_mat
    var_name = varname
    tp_name = tpName
    csv_name = tp_name + '.csv'
    matfile_name = tp_name + '.mat'

    # load data
    # X_df = load_data_csv(DIR_PATH_csv, 'X_processed_' + csv_name)
    # Y_df = load_data_csv(DIR_PATH_csv, 'Y_processed_' + csv_name)
    X_df = load_data_csv(DIR_PATH_csv, 'X_' + var_name + csv_name)
    Y_df = load_data_csv(DIR_PATH_csv, 'Y_' + var_name + csv_name)
    mu = load_data_csv(DIR_PATH_csv, 'mu_' + var_name + csv_name)
    sigma = load_data_csv(DIR_PATH_csv, 'sigma_' + var_name + csv_name)

    tradedates_df = Y_df[['date']]
    X_df = X_df.drop('tradedate', axis=1)
    Y_df = Y_df.drop('date', axis=1)

    X_ =X_df.values
    Y_ =Y_df.values
    tradedates_ = tradedates_df.values
    mu = mu.values[0][0]
    sigma = sigma.values[0][0]

    sequenceSize = load_data_mat(DIR_PATH_mat, matfile_name, 'SequenceSize')[0][0]
    train_miniBatchMultiple = load_data_mat(DIR_PATH_mat, matfile_name, 'train_miniBatchMultiple')[0][0]
    train_validBatchMultiple = load_data_mat(DIR_PATH_mat, matfile_name, 'valid_batchMultiple')[0][0]
    batchSize_multiple = load_data_mat(DIR_PATH_mat, matfile_name, 'batchSize_multiple')[0][0]

    # define mini batch size and validation set size
    if not MinBatchMultiple and MinBatchMultiple !=0:
        MinBatchSize = train_miniBatchMultiple * batchSize_multiple
    else:
        MinBatchSize = MinBatchMultiple * batchSize_multiple

    if not ValidBatchMultiple and ValidBatchMultiple != 0:
        ValidSize = train_validBatchMultiple * batchSize_multiple # use default (from .mat file)
    else:
        ValidSize = ValidBatchMultiple * batchSize_multiple

    # pad in zeros to make number of features a multiple of 8 -- for mixed precision training
    # padding = np.zeros([len(X), 6])
    # X = np.concatenate((X, padding), axis=1)

    # produce sequence
    look_back = sequenceSize  # multiple of 8 for float16 training
    num_features = X_.shape[1]
    num_samples = X_.shape[0] - look_back + 1

    X_sequence = np.zeros((num_samples, look_back, num_features))
    Y_sequence = np.zeros((num_samples))
    tradedates_sequence = np.empty((num_samples), dtype='object')

    for i in range(num_samples):
        Y_position = i + look_back
        X_sequence[i] = X_[i:Y_position]
        Y_sequence[i] = Y_[Y_position - 1]
        tradedates_sequence[i] = tradedates_[Y_position - 1]

    print("X_sequence: " + str(X_sequence.shape))
    print("Y_sequence: " + str(Y_sequence.shape))

    X = X_sequence
    Y = Y_sequence
    flat_list = []
    for sublist in tradedates_sequence:
        for item in sublist:
            flat_list.append(item)
    tdates = np.asarray(flat_list)

    # remove outlier dates FROM HERE-------------------------------------------------
    # % remove
    # outlier
    # dates
    # if isequal(Asset, 'KOSPI')
    # if ~exist('outlierDates_KOSPI', 'var')
    # error('Global variable ''outlierDates_KOSPI'' does not exist. ')
    # end
    # outlierDates = outlierDates_KOSPI;
    # end
    # if isequal(Asset, 'KGB')
    # if ~exist('outlierDates_KGB', 'var')
    # error('Global variable ''outlierDates_KGB'' does not exist.')
    # end
    # outlierDates = outlierDates_KGB;
    # end
    # OLidx = ismember(Y.(1), outlierDates );
    # Y = Y(~OLidx,:);
    # X = X(~OLidx,:);  TO HERE ----------------------------------------------------

    # divide training, validation and test sets
    # slice off test dataset
    if TestSize == 0:
        X_test = np.empty([0])
        X = X
        Y_test = np.empty([0])
        Y = Y
        tdates_test = pd.DataFrame()
        tdates = tdates
    else:
        X_test = X[TestSize * -1:]
        X = X[:TestSize * -1]
        Y_test = Y[TestSize * -1:]
        Y = Y[:TestSize * -1]
        tdates_test = tdates[TestSize * -1:]
        tdates = tdates[:TestSize * -1]

    # cut off to multiple of batchSize_multiple
    if len(Y) % batchSize_multiple != 0:
        beginIDX = batchSize_multiple * math.floor(len(Y) / batchSize_multiple)
        X = X[beginIDX * -1:]
        Y = Y[beginIDX * -1:]
        tdates = tdates[beginIDX * -1:]

    # cutoff to include most recent observations
    if len(Y) % MinBatchSize != 0:
        beginIDX = MinBatchSize * math.floor(len(Y) / MinBatchSize)
        X = X[beginIDX * -1:]
        Y = Y[beginIDX * -1:]
        tdates = tdates[beginIDX * -1:]

    # throw error if lengths don't match
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y lengths differ')
    if Y.shape[0] != tdates.shape[0]:
        raise ValueError('Y and tdates lengths differ')

    # slice off validation dataset
    if ValidSize == 0:
        Y_train = Y
        X_train = X
        tdates_train = tdates
        Y_valid = np.empty([0])
        X_valid = np.empty([0])
        tdates_valid = np.empty([0])
    else:
        Y_train = Y[:len(Y)-ValidSize]
        X_train = X[:len(X)-ValidSize]
        tdates_train = tdates[:len(tdates)-ValidSize]
        Y_valid = Y[ValidSize * -1:]
        X_valid = X[ValidSize * -1:]
        tdates_valid = tdates[ValidSize * -1:]

    # print out data shapes
    print("Train feature shape: " + str(X_train.shape))
    print("Train target shape: " + str(Y_train.shape))
    print("Train dates: " + str(tdates_train.shape))
    print("Validation feature shape: " + str(X_valid.shape))
    print("Validation target shape: " + str(Y_valid.shape))
    print("Validation dates: " + str(tdates_valid.shape))

    if X_test.any():
        print("Test feature shape: " + str(X_test.shape))
        print("Test target shape: " + str(Y_test.shape))
        print("Test dates: " + str(tdates_test.shape))
    else:
        print("Test feature shape: empty")
        print("Test target shape: empty")
        print("Test dates: empty")

    # output
    tradedates = {"tradedates_train":tdates_train, "tradedates_valid":tdates_valid, "tradedates_test":tdates_test}
    output_dict = {"X_train":X_train, "Y_train":Y_train, "X_valid":X_valid, "Y_valid":Y_valid,
                   "X_test": X_test, "Y_test": Y_test, "batchSize_multiple":batchSize_multiple,
                   "tradedates":tradedates, "mu":mu, "sigma":sigma}

    return output_dict


def Import3Ddata(path_mat, tpName, varname='', MinBatchMultiple=0, ValidBatchMultiple=0, TestSize=0):

    # import packages
    import os
    import scipy.io as spio
    import math
    import numpy as np

    # define function
    def load_data_mat(dir_path, sub_path, string):
        matfile_path = os.path.join(dir_path, sub_path)
        var = spio.loadmat(matfile_path, variable_names=string, chars_as_strings=True)
        return var[string]

    # set path variables
    DIR_PATH_mat = path_mat
    var_name = varname
    tp_name = tpName
    matfile_name = tp_name + var_name + '.mat'

    # load data
    X = load_data_mat(DIR_PATH_mat, matfile_name, 'Xarray')
    Y = load_data_mat(DIR_PATH_mat, matfile_name, 'Yarray')

    mu = load_data_mat(DIR_PATH_mat, matfile_name, 'mu')[0][0]
    sigma = load_data_mat(DIR_PATH_mat, matfile_name, 'sigma')[0][0]
    train_miniBatchMultiple = load_data_mat(DIR_PATH_mat, matfile_name, 'train_miniBatchMultiple')[0][0]
    train_validBatchMultiple = load_data_mat(DIR_PATH_mat, matfile_name, 'valid_batchMultiple')[0][0]
    batchSize_multiple = load_data_mat(DIR_PATH_mat, matfile_name, 'batchSize_multiple')[0][0]
    tdates = load_data_mat(DIR_PATH_mat, matfile_name, 'tradedates')

    # X = tf.cast(X, tf.float32)
    # Y = tf.cast(Y, tf.float32)

    print("Feature shape: " + str(X.shape))
    print("Target shape: " + str(Y.shape))

    # define mini batch size and validation set size
    if not MinBatchMultiple and MinBatchMultiple !=0:
        MinBatchSize = train_miniBatchMultiple * batchSize_multiple
    else:
        MinBatchSize = MinBatchMultiple * batchSize_multiple

    if not ValidBatchMultiple and ValidBatchMultiple != 0:
        ValidSize = train_validBatchMultiple * batchSize_multiple
    else:
        ValidSize = ValidBatchMultiple * batchSize_multiple

    # divide training, validation and test sets
    # slice off test dataset
    if TestSize == 0:
        X_test = np.empty([0])
        X = X
        Y_test = np.empty([0])
        Y = Y
        tdates_test = pd.DataFrame()
        tdates = tdates
    else:
        X_test = X[TestSize * -1:]
        X = X[:TestSize * -1]
        Y_test = Y[TestSize * -1:]
        Y = Y[:TestSize * -1]
        tdates_test = tdates[TestSize * -1:]
        tdates = tdates[:TestSize * -1]

    # cut off to multiple of batchSize_multiple
    if len(Y) % batchSize_multiple != 0:
        beginIDX = batchSize_multiple * math.floor(len(Y) / batchSize_multiple)
        X = X[beginIDX * -1:]
        Y = Y[beginIDX * -1:]
        tdates = tdates[beginIDX * -1:]

    # cutoff to include most recent observations
    if len(Y) % MinBatchSize != 0:
        beginIDX = MinBatchSize * math.floor(len(Y) / MinBatchSize)
        X = X[beginIDX * -1:]
        Y = Y[beginIDX * -1:]
        tdates = tdates[beginIDX * -1:]

    # throw error if lengths don't match
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y lengths differ')
    if Y.shape[0] != tdates.shape[0]:
        raise ValueError('Y and tdates lengths differ')

    # slice off validation dataset
    if ValidSize == 0:
        Y_train = Y
        X_train = X
        tdates_train = tdates
        Y_valid = np.empty([0])
        X_valid = np.empty([0])
        tdates_valid = np.empty([0])
    else:
        Y_train = Y[:len(Y)-ValidSize]
        X_train = X[:len(X)-ValidSize]
        tdates_train = tdates[:len(tdates)-ValidSize]
        Y_valid = Y[ValidSize * -1:]
        X_valid = X[ValidSize * -1:]
        tdates_valid = tdates[ValidSize * -1:]

    # print out data shapes
    print("Train feature shape: " + str(X_train.shape))
    print("Train target shape: " + str(Y_train.shape))
    print("Train dates: " + str(tdates_train.shape))
    print("Validation feature shape: " + str(X_valid.shape))
    print("Validation target shape: " + str(Y_valid.shape))
    print("Validation dates: " + str(tdates_valid.shape))

    if X_test.any():
        print("Test feature shape: " + str(X_test.shape))
        print("Test target shape: " + str(Y_test.shape))
        print("Test dates: " + str(tdates_test.shape))
    else:
        print("Test feature shape: empty")
        print("Test target shape: empty")
        print("Test dates: empty")

    # output
    tradedates = {"tadedates_train":tdates_train, "tradedates_valid":tdates_valid, "tradedates_test":tdates_test}
    output_dict = {"X_train":X_train, "Y_train":Y_train, "X_valid":X_valid, "Y_valid":Y_valid,
                   "X_test": X_test, "Y_test": Y_test, "batchSize_multiple":batchSize_multiple,
                   "tradedates":tradedates, "mu":mu, "sigma":sigma}

    return output_dict



def ImportCSVtable_inLSTM_sequence(path_CSV_feature, path_CSV_target, sequenceSize, batchSize_multiple=8,
                                   MinBatch_multiple=3, Validation_multiple=0, TestSize=0, leap=1,
                                   plot_histogram=False, plot_heatmap=False):

    # path_CSV_feature = r'C:\Users\jp\Google Drive\MATLAB data files\Commercialize\LSTM MD strat\feat_OC_b7f4.csv'
    # path_CSV_target = r'C:\Users\jp\Google Drive\MATLAB data files\Commercialize\LSTM MD strat\tgt_OC_b7f4.csv'
    # sequenceSize = 96
    # batchSize_multiple = 8
    # MinBatch_multiple = 3
    # Validation_multiple = 0
    # TestSize = 0


    # import packages
    import numpy as np
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()

    # load data
    X_df = pd.read_csv(path_CSV_feature)
    Y_df = pd.read_csv(path_CSV_target)

    tradedates_df = Y_df[['date']]
    X_df = X_df.drop(X_df.columns[0], axis=1)
    Y_df = Y_df.drop(Y_df.columns[0], axis=1)

    # plot histogram
    if plot_histogram:
        X_df.hist(bins=50, figsize=(20, 15))
        plt.tight_layout()
        print("X raw")
        plt.show()

        Y_df.hist(bins=50)
        plt.tight_layout()
        plt.title("Y raw")
        plt.show()

    # standardize
    from sklearn.preprocessing import StandardScaler

    X_ = X_df.values
    Y_ = Y_df.values
    tradedates_ = tradedates_df.values

    scaler_std = StandardScaler().fit(X_)
    X_ = scaler_std.transform(X_)

    mu = Y_.mean()
    sigma = Y_.std()
    Y_ = ( Y_ - mu ) / sigma

    # plot histogram again (after standardizing)
    if plot_histogram:
        X_df.hist(bins=50, figsize=(20, 15))
        plt.tight_layout()
        print("X standardized (not clipped)")
        plt.show()

        Y_df.hist(bins=50)
        plt.tight_layout()
        plt.title("Y standardized (not clipped)")
        plt.show()

    # plot heatmap
    if plot_heatmap:
        ax = sns.heatmap(X_)
        plt.title("feature hetmap")
        plt.show()

    # produce sequence
    look_back = sequenceSize  # multiple of 8 for float16 training
    num_features = X_.shape[1]
    num_samples = X_.shape[0] - look_back + 1

    X_sequence = np.zeros((num_samples, look_back, num_features))
    Y_sequence = np.zeros((num_samples))
    tradedates_sequence = np.empty((num_samples), dtype='object')

    for i in range(num_samples):
        Y_position = i + look_back
        X_sequence[i] = X_[i:Y_position]
        Y_sequence[i] = Y_[Y_position - 1]
        tradedates_sequence[i] = tradedates_[Y_position - 1]

    print("X_sequence: " + str(X_sequence.shape))
    print("Y_sequence: " + str(Y_sequence.shape))

    X = X_sequence
    Y = Y_sequence
    flat_list = []
    for sublist in tradedates_sequence:
        for item in sublist:
            flat_list.append(item)
    tdates = np.asarray(flat_list)
    print("tdates: " + str(tdates.shape))

    # divide training, validation and test sets
    # slice off test dataset
    MinBatchSize = batchSize_multiple * MinBatch_multiple
    if TestSize == 0:
        X_test = np.empty([0])
        X = X
        Y_test = np.empty([0])
        Y = Y
        tdates_test = pd.DataFrame()
        tdates = tdates
    else:
        X_test = X[TestSize * -1:]
        X = X[:TestSize * -1]
        Y_test = Y[TestSize * -1:]
        Y = Y[:TestSize * -1]
        tdates_test = tdates[TestSize * -1:]
        tdates = tdates[:TestSize * -1]

    # slice off validation dataset
    ValidationSize = batchSize_multiple * Validation_multiple
    if ValidationSize == 0:
        Y_valid = np.empty([0])
        X_valid = np.empty([0])
        tdates_valid = np.empty([0])

        Y_train = Y
        X_train = X
        tdates_train = tdates
    else:
        Y_valid = Y[ValidationSize * -1:]
        X_valid = X[ValidationSize * -1:]
        tdates_valid = tdates[ValidationSize * -1:]

        Y_train = Y[:len(Y)-ValidationSize]
        X_train = X[:len(X)-ValidationSize]
        tdates_train = tdates[:len(tdates)-ValidationSize]

        # apply leap (1 is for no leap)
        if leap < 1:
            raise ValueError('leap value must be 1 or larger')
        elif leap == 1:
            X_train = X_train
            Y_train = Y_train
        elif leap > 1:
            X_train = X_train[:(leap - 1) * -1]
            Y_train = Y_train[:(leap - 1) * -1]
            tdates_train = tdates_train[:(leap - 1) * -1]

    # cutoff to include most recent observations in train set
    if len(Y_train) % MinBatchSize != 0:
        beginIDX = MinBatchSize * math.floor(len(Y_train) / MinBatchSize)
        X_train = X_train[beginIDX * -1:]
        Y_train = Y_train[beginIDX * -1:]
        tdates_train = tdates_train[beginIDX * -1:]

    # throw error if lengths don't match
    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError('X_train and Y_train lengths differ')
    if Y_train.shape[0] != tdates_train.shape[0]:
        raise ValueError('Y_train and tdates_train lengths differ')

    # print out data shapes
    print("Train feature shape: " + str(X_train.shape))
    print("Train target shape: " + str(Y_train.shape))
    print("Train dates: " + str(tdates_train.shape))
    print("Validation feature shape: " + str(X_valid.shape))
    print("Validation target shape: " + str(Y_valid.shape))
    print("Validation dates: " + str(tdates_valid.shape))

    if X_test.any():
        print("Test feature shape: " + str(X_test.shape))
        print("Test target shape: " + str(Y_test.shape))
        print("Test dates: " + str(tdates_test.shape))
    else:
        print("Test feature shape: empty")
        print("Test target shape: empty")
        print("Test dates: empty")

    # output
    tradedates = {"tradedates_train": tdates_train, "tradedates_valid": tdates_valid, "tradedates_test": tdates_test}
    output_dict = {"X_train":X_train, "Y_train":Y_train, "X_valid":X_valid, "Y_valid":Y_valid,
                   "X_test": X_test, "Y_test": Y_test, "batchSize_multiple":batchSize_multiple,
                   "tradedates":tradedates, "mu":mu, "sigma":sigma}

    return output_dict
