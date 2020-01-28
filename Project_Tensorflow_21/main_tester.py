# %% LOOPS THROUGH PARAMETERS WITH 1 DATASET

# %% import base packages

import packages_utilities
packages_utilities.select_gpu("DESKTOP-DNL8QNB")

import os
pc_name = os.environ['COMPUTERNAME']

import tensorflow as tf
import numpy as np
# import math
# import operator
import datetime
import csv
# import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
# from scipy import io as spio
import shutil
import pickle

import time
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import seaborn as sbn

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers, regularizers, losses
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# from import_data import Import3Ddata
from packages_training import r_square, propAcc, create_model, \
    create_movMetricStop_callback, cb_PrintDot, selectmodel_basedon_movMetric
# import sys

print(tf.__version__)
print(tf.keras.__version__)

# %% check GPU & control gpu memory usage -- not necesary for windows 10 (?)

# tf.debugging.set_log_device_placement(True) # shows which device is being used

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth option is set")
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # define timestamp for this run

# %% set file paths & import data (for day trading strat)

# Linux--------------------------------------------------------------------------------------------------
# path_CSV = '/home/lstm/Desktop/GoogleDrive_local/Data Release/CSV'
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files 8 multiple 2 dim' # Linux
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files 8 multiple all dim'
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files temp'


# 8700k--------------------------------------------------------------------------------------------------
if pc_name == "DESKTOP-DNL8QNB":
    path_CSV = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV_SD' # 8700k
    #path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro/Quant Project\Outsourcing/Data Release\mat files 8 multiple all dim'
    path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files 8 multiple 2 dim'
    #path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files temp'

    checkpoint_dir = r'C:\Users\jp\TF_checkpoints_Tests' # checkpoint weights
    logdir_rootroot = r'C:\Users\jp\Google Drive\TF_logs\Logs_Tests\\'+ pc_name+'\\' # for tensorboard
    saveDIR_root = r'C:\Users\jp\Google Drive\TF_Predictions\Predictions_Tests\\'+ pc_name # for CSV

# Windows on Unbuntu01-----------------------------------------------------------------------------------
if pc_name == "DESKTOP-U9PDCSR":
    path_CSV = r'C:\Users\jp\Google Drive\Data Release\CSV_SD'
    # path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple all dim'
    path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple 2 dim'
    # path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files temp'

    checkpoint_dir = r'F:\TF_checkpoints_Tests' # checkpoint weights
    logdir_rootroot = r'C:\Users\jp\Google Drive\TF_logs\Logs_Tests\\'+ pc_name+'\\' # for tensorboard
    saveDIR_root = r'C:\Users\jp\Google Drive\TF_Predictions\Predictions_Tests\\'+ pc_name # for CSV

# import data --------------------------------------------------------------------------------------------
# tpName = 'KOSPI_HLmid_mrng1030'
tpName = 'KOSPI_HLmid_mrng'
# tpName = 'KOSPI_PTTP_mrng1030'
# tpName = 'KOSPI_PTTP_mrng'
# tpName = 'KOSPI_HIGH_mrng1030'
# tpName = 'KOSPI_LOW_mrng1030'

# varname = ''                      # for importing processed, clipped 3D data
varname = 'processed_'            # for importing processed, clipped CSV data
# varname = 'unprocessed_'          # for importing unprocessed CSV data
# varname = 'processed_unclipped_'    # for importing processed, unclipped CSV data
# varname = 'processed_clipped25'   # for importing processed CSV data, clipped at 25

test_size_range = [0, 24, 48, 72]
MinBatchMultiple = 3 # mutiple of batchSize_multiple (usually 8); use default if 0
ValidBatchMultiple = 3 # must be multiple of MinBatchMultiple; use default if 0
data_dict = {}

for i in range(len(test_size_range)):
    TestSize = test_size_range[i]
    print(TestSize)

    # from import_data import Import3Ddata
    # imported_ = Import3Ddata(path_mat, tpName, varname, MinBatchMultiple, ValidBatchMultiple, TestSize)

    from import_data import ImportCSVdata_processed
    imported_ = ImportCSVdata_processed(path_CSV, path_mat, tpName, varname,
                                        MinBatchMultiple, ValidBatchMultiple, TestSize)
    data_dict[i] = imported_

# %% If testing MD strat, rE-define file paths & re-import data (based on 8700k path)

path_CSV = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV_MD_select'
feat_nums = 'select_' # 'select_' or 'all_'
# path_CSV = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV_MD_all'
# feat_nums = 'all_' # 'select_' or 'all_'

asset = 'KGB'
backshift = '7'
forwardshift = '4'
if forwardshift == 0:
    fshift_suffix = '_new'
else:
    fshift_suffix = ''

if pc_name == "DESKTOP-DNL8QNB":

    path_CSV_feature = path_CSV+r'\\X_MD_'+feat_nums+asset+'_b'+backshift+'_f'+forwardshift+fshift_suffix+'.csv'
    path_CSV_target = path_CSV+r'\\Y_MD_'+feat_nums+asset+'_b'+backshift+'_f'+forwardshift+fshift_suffix+'.csv'

else: raise ValueError('MD strat file paths are currently set for 8700k only')

tpName = 'KGB_MDstrat'
sequenceSize = 96 # this is a major parameter to try different values on
batchSize_multiple = 8
MinBatchMultiple = 3
Validation_multiple = 3
TestSize = 0 # leave at 0 for main_tester
plot_histogram = False
plot_heatmap = True

test_size_range = [0, 24, 48, 72]
data_dict = {}

for i in range(len(test_size_range)):
    TestSize = test_size_range[i]
    print(TestSize)

    # from import_data import Import3Ddata
    # imported_ = Import3Ddata(path_mat, tpName, varname, MinBatchMultiple, ValidBatchMultiple, TestSize)

    from import_data import ImportCSVtable_inLSTM_sequence
    imported_ = ImportCSVtable_inLSTM_sequence(path_CSV_feature, path_CSV_target, sequenceSize, batchSize_multiple,
                                   MinBatchMultiple, Validation_multiple, TestSize, 4, plot_histogram, plot_heatmap)
    data_dict[i] = imported_

# %% define params

num_filters = 108
filter_szie = 7
num_days_4_convLSTM = 7 # days to include in ConvLSTM2D convolution
num_channels = 1 # number of channels for convolution --> value is 1 except in extraordinary cases
num_features = data_dict[0]["X_train"].shape[2]
if not 'sequenceSize' in locals():
    sequenceSize = 96

bi_dir = False

conv_1D = False
if conv_1D:
    conv_params = {'filters':num_filters, 'kernel':1, 'activation':'relu', 'padding':'same'}
    conv_input_shape = [sequenceSize, num_features]
    CONVsize=2

conv_2D = False
if conv_2D:
    conv_params = {'filters':num_filters, 'kernel':filter_szie, 'activation':'relu', 'padding':'same'}
    conv_input_shape = [sequenceSize, num_features, num_channels]
    CONVsize=4

conv_LSTM_2D = False
if conv_LSTM_2D:
    conv_params = {'filters':num_filters, 'kernel':filter_szie, 'activation':'relu', 'padding':'same'}
    conv_input_shape = [batchSize_multiple*MinBatchMultiple, num_days_4_convLSTM,
                        sequenceSize, num_features, num_channels]

if not conv_1D and not conv_2D and not conv_LSTM_2D:
    CONVsize=[]
    conv_input_shape=[]
    conv_params = {}


# abridged_validationSize = 15 # days to validate historical models for selection of current model

# early_stop_metric = "R2 rolling mean"
mov_windowsSize = 5  # applicable only if rolling mean is used
early_stop_metric = "single metric"

train_optimizer = [
#     [optimizers.Adadelta(clipnorm=1.),'_vanila'],
#     [optimizers.SGD(clipnorm=1.),'_vanila'],
#     [optimizers.SGD(clipnorm=1., momentum=0., nesterov=True), '_Nstrv'],
#     [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=False), '_m05_Nstrv'],
#     [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=True), '_m05'],
#     [optimizers.SGD(lr=0.01, clipnorm=1., momentum=0.9, nesterov=True), '_m09_lr_0-01_Nstrv'],
#     [optimizers.SGD(lr=0.1, clipnorm=1., momentum=0.9, nesterov=True), '_m09_lr_0-1_Nstrv'],
#     [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=False), '_m09'],
#     [optimizers.RMSprop(clipnorm=1.),'_vanila'],
#     [optimizers.Adagrad(clipnorm=1.),'_vanila'],
#     [optimizers.Adamax(clipnorm=1.),'_vanila'],
#     [optimizers.Nadam(lr=0.0005, clipnorm=1.), '_lr0-0005'],
#     [optimizers.Nadam(lr=0.001, clipnorm=1.), '_lr0-001'],
    [optimizers.Nadam(lr=0.002, clipnorm=1.), '_lr0-002']
    # [optimizers.Nadam(lr=0.005, clipnorm=1.), '_lr0-005'],
    # [optimizers.Nadam(lr=0.01, clipnorm=1.), '_lr0-01']
#     [optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False, clipnorm=1.),'_vanila']
]
train_activation = [
    # tf.nn.leaky_relu,
    # tf.nn.relu,
    # tf.nn.relu6,
    # tf.nn.selu,
    tf.nn.elu
]
train_lossfunc = [
    losses.MeanSquaredError(),
    # losses.MeanAbsoluteError(),
    # losses.Huber(delta=0.01),
    # losses.Huber(delta=0.05),
    # losses.Huber(delta=0.1),
    # losses.Huber(delta=0.3),
    # losses.Huber(delta=0.6)
    # losses.Huber(delta=1.0),
    # losses.Huber(delta=2.0),
    # losses.Huber(delta=5.0),
    # losses.Huber(delta=10.0),
    # losses.LogCosh(),
    # losses.MeanAbsolutePercentageError(),
    # losses.MeanSquaredLogarithmicError(),
    # losses.CosineSimilarity(),
    # losses.Poisson(),
    # losses.BinaryCrossentropy(),
    # losses.CategoricalCrossentropy(),
    # losses.CategoricalHinge(),
    # losses.Hinge(),
    # losses.KLDivergence(),
    # losses.LogLoss(),
    # losses.SparseCategoricalCrossentropy(),
    # losses.SquaredHinge()
]
LSTMsize = [
    # [720, 720],
    # [720, 480],
    # [480, 480],
    # [480, 240],
    # [240, 240],
    [240, 120],
    # [120, 120],
    # [120, 60],
    # [60, 60],
    # [60, 30]
    # [30, 30]
]
FCsize = [
    #[128],
    [data_dict[0]['X_train'].shape[2]]
]

train_LSTMinputDroprate = 0.5
train_LSTMrecDroprate = 0
train_droprate = 0.5
train_l2reg = 0.0001

num_epoch = 3000
min_epochs_ = 15
patience_= 500
output_unit = 1

run_identifier = ''
logdir_root=logdir_rootroot+'_'+tpName+'_'+run_identifier+'_'+time_stamp

if not os.path.exists(saveDIR_root):
    os.mkdir(saveDIR_root)

savepath_preds_csv = saveDIR_root + '\\'+ tpName+'_'+run_identifier + '_' + time_stamp + '_preds.csv' # overwrite predictions on same file
savepath_config = saveDIR_root + '\\'+ tpName+'_'+run_identifier + '_' + time_stamp + '_config.pkl' # path for model configuration
savepath_models = saveDIR_root + '\\'+ tpName+'_'+run_identifier+'_' + time_stamp+'_models'
savepath_imgs = saveDIR_root + '\\'+ tpName+'_'+run_identifier+'_' + time_stamp+'_images'
savepath_metrics = saveDIR_root + '\\'+ tpName+'_'+run_identifier+'_' + time_stamp+'_metric.csv'

# %% setup callbacks (the parameters of which do not depend on runs)

checkpoint_modifier = '\cp-{epoch:04d}'
checkpoint_filesuffix = '.hdf5'
checkpoint_path = checkpoint_dir + checkpoint_modifier + checkpoint_filesuffix
if early_stop_metric == "single metric":
    bestonly = True
else:
    bestonly = False

# %% get iterable object (combination of all parameters)

loop_obj ={}
counter = 0
for i in range(len(train_optimizer)):
    train_optimizer_ = train_optimizer[i][0]
    optimizer_name = train_optimizer_.__dict__.get("_name") + train_optimizer[i][1]

    for train_activation_ in train_activation:
        activation_name = train_activation_.__name__

        for train_lossfunc_ in train_lossfunc:
            lossfunc_name = train_lossfunc_.__dict__.get("name")
            if lossfunc_name == 'huber_loss':
                lossfunc_name = lossfunc_name+'_d'+str(train_lossfunc_.get_config()['delta'])

            for LSTMsize_ in LSTMsize:
                LSTMname = 'LSTMsize_' + '_'.join(map(str, LSTMsize_))

                for FCsize_ in FCsize:
                    FCname = 'FCsize_' + '_'.join(map(str, FCsize_))

                    for key in data_dict:
                        data_name = key

                        loop_obj_sub = {'optimizer':[train_optimizer_,optimizer_name],
                                        'activation':[train_activation_, activation_name],
                                        'lossfunc':[train_lossfunc_, lossfunc_name],
                                        'LSTM':[LSTMsize_, LSTMname],
                                        'FC':[FCsize_, FCname],
                                        'data_key': data_name}
                        loop_obj[counter] = loop_obj_sub
                        counter += 1

print('Run: tensorboard --logdir "'+logdir_root+'"')
input("Press Enter to continue...")

# %% run in loop

print("\nRunning loop with following parameters:")
print('optimizers: '+str(len(train_optimizer)))
print('activations: '+str(len(train_activation)))
print('loss functions: '+str(len(train_lossfunc)))
print('LSTM structures: '+str(len(LSTMsize)))
print('FC structures: '+str(len(FCsize)))
print('datasets: '+str(len(data_dict)))
print('Altogether '+str(len(loop_obj))+' iterations')
input('Continue?')

for i in loop_obj:
    current_obj = loop_obj[i]

    # define parameters for iteration------------------------------------------------------------------------
    train_optimizer_ = current_obj['optimizer'][0]
    optimizer_name = current_obj['optimizer'][1]

    train_activation_ = current_obj['activation'][0]
    activation_name = current_obj['activation'][1]

    train_lossfunc_ = current_obj['lossfunc'][0]
    lossfunc_name = current_obj['lossfunc'][1]

    LSTMsize_ = current_obj['LSTM'][0]
    LSTMname = current_obj['LSTM'][1]

    FCsize_ = current_obj['FC'][0]
    FCname = current_obj['FC'][1]

    Data_ = data_dict[current_obj['data_key']]
    DataName = 'Data_'+str(current_obj['data_key'])

    # get unique name for iteration [BUG: hinges losses and SCC don't have name]----------------------------
    # print(i)
    iteration_id = optimizer_name + '_' + activation_name + '_' + lossfunc_name + '_' + LSTMname + '_' + FCname + '_' + DataName
    print(iteration_id)

    # refresh checkpoint folder------------------------------------------------------------------------------
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        time.sleep(3)
        while not os.path.isdir(checkpoint_dir):
            print('Creating ' + checkpoint_dir)
            time.sleep(3)
            os.makedirs(checkpoint_dir)
        print(checkpoint_dir + ' created')

    # define data-------------------------------------------------------------------------------------------
    X_train = data_dict[current_obj['data_key']]["X_train"]
    Y_train = data_dict[current_obj['data_key']]["Y_train"]
    X_valid = data_dict[current_obj['data_key']]["X_valid"]
    Y_valid = data_dict[current_obj['data_key']]["Y_valid"]
    X_test = data_dict[current_obj['data_key']]["X_test"]
    Y_test = data_dict[current_obj['data_key']]["Y_test"]
    batchSize_multiple = data_dict[current_obj['data_key']]["batchSize_multiple"]
    tdates_train = data_dict[current_obj['data_key']]['tradedates']["tradedates_train"]
    tdates_valid = data_dict[current_obj['data_key']]['tradedates']["tradedates_valid"]
    tdates_test = data_dict[current_obj['data_key']]['tradedates']["tradedates_test"]
    mu = imported_["mu"]
    sigma = imported_["sigma"]

    print('X_train dimension: ' + str(X_train.shape))
    print('Y_train dimension: ' + str(Y_train.shape))
    print('X_valid dimension: ' + str(X_valid.shape))
    print('Y_valid dimension: ' + str(Y_valid.shape))
    print('X_test dimension: ' + str(X_test.shape))
    print('tdates_test dimension: ' + str(tdates_test.shape))

    MinBatchSize = MinBatchMultiple * batchSize_multiple
    Xdim = X_train.shape
    minibatch_shape= [MinBatchSize, Xdim[1], Xdim[2]]

    # setup callbacks -----------------------------------------------------------------------------------------
    logdir_run=r'\\' + iteration_id
    cb_tensorboard = TensorBoard(log_dir=logdir_root + logdir_run, histogram_freq=0)

    if early_stop_metric == "single metric":
        cb_trainstop = EarlyStopping(monitor='val_r_square', patience=patience_,
                                mode='max', restore_best_weights=True)
    elif early_stop_metric == "R2 rolling mean":
        cb_trainstop = create_movMetricStop_callback(stop_patience=patience_,
                                            min_epochs=min_epochs_, logdir=logdir_root+logdir_run)
    else:
        raise NameError('early_stop_metric not defined')

    cb_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=bestonly, save_weights_only=True,
                                    period=1, verbose=0)

    CBs = [cb_tensorboard, cb_checkpoint, cb_trainstop, cb_PrintDot()]

    # set up mixed precision training------------------------------------------------------------------------
    # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    # tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
    # tf.keras.mixed_precision.experimental.set_policy('infer')
    # print(tf.keras.mixed_precision.experimental.global_policy())

    # X_train = tf.cast(X_train, tf.float32)
    # Y_train = tf.cast(Y_train, tf.float32)
    # X_valid = tf.cast(X_valid, tf.float32)
    # Y_valid = tf.cast(Y_valid, tf.float32)

    # set up distributed deivces-----------------------------------------------------------------------------
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

    # create & run model-------------------------------------------------------------------------------------
    # with tf.device("/device:GPU:0"):

    if conv_2D:
        X_train = np.expand_dims(X_train,axis=3)
        X_valid = np.expand_dims(X_valid,axis=3)
        X_test = np.expand_dims(X_test,axis=3)

    model = create_model(minibatch_shape,
                           LSTMsize_,
                           FCsize_,
                           CONVsize,
                           train_optimizer_,
                           activationF=train_activation_,
                           LSTMinputDroprate=train_LSTMinputDroprate,
                           LSTMrecDroprate=train_LSTMrecDroprate,
                           droprate=train_droprate,
                           l2reg=train_l2reg,
                           lossfunc=train_lossfunc_,
                           bi_dir=bi_dir,
                           conv_1D = conv_1D,
                           conv_2D = conv_2D,
                           conv_LSTM_2D=conv_LSTM_2D,
                           conv_params=conv_params,
                           conv_input_shape = conv_input_shape,
                           compile_model=True
                           )

    history = model.fit(X_train, Y_train,
                        validation_data=(X_valid, Y_valid),
                        batch_size=MinBatchSize,
                        epochs=num_epoch,
                        verbose=0,
                        callbacks=CBs,
                        shuffle=False
                        )

    # select model from checkpoints------------------------------------------------------------
    if early_stop_metric == "single metric":
        model_selected =  model
        selected_epoch = os.listdir(checkpoint_dir)[-1]
    elif early_stop_metric == "R2 rolling mean":
        model_selected, selected_epoch = selectmodel_basedon_movMetric(X_valid, Y_valid, mov_windowsSize,
                    history, checkpoint_dir, checkpoint_filesuffix, model, metric='val_r_square')
    else:
        raise NameError('early_stop_metric not defined')

    # make prediction with selected model on validation dataset----------------------------------------
    preds_valid = model_selected.predict(X_valid)
    preds_valid = preds_valid * sigma + mu
    preds_valid = preds_valid.reshape([len(preds_valid), 1])
    preds_tdates_valid = tdates_valid
    preds_ground_valid = np.array(Y_valid)
    preds_ground_valid = preds_ground_valid * sigma + mu
    preds_ground_valid = preds_ground_valid.reshape([len(preds_ground_valid), 1])
    r2_valid = r2_score(preds_ground_valid, preds_valid)

    acc_valid = accuracy_score(np.sign(preds_ground_valid), np.sign(preds_valid))

    # make prediction with selected model on test dataset----------------------------------------
    if X_test.any():

        preds = model_selected.predict(X_test)
        preds = preds * sigma + mu
        preds = preds.reshape([len(preds),1])
        preds_tdates = tdates_test
        preds_ground = np.array(Y_test)
        preds_ground = preds_ground * sigma + mu
        preds_ground = preds_ground.reshape([len(preds_ground), 1])
        r2_test = r2_score(preds_ground, preds)

        acc_test = accuracy_score(np.sign(preds_ground), np.sign(preds))

        # plots--------------------------------------------------------------------------------------
        plt.figure(figsize=[8.27,11.69])
        plt.subplot(311)
        plt.plot(history.history['r_square'])
        plt.plot(history.history['val_r_square'])
        plt.title(optimizer_name+'_'+activation_name+'_'+lossfunc_name+'_'+LSTMname+'_'+FCname+', '+preds_tdates[0])
        plt.ylabel('R^2')
        # plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.subplot(312)
        xx_axis = list(range(1, len(preds_tdates_valid) + 1))
        plt.xticks(xx_axis, preds_tdates_valid, rotation=70)
        plt.stem(xx_axis, preds_valid, '.', 'x')
        plt.stem(xx_axis, preds_ground_valid, '.', 'o')
        r2_valid_ = "{:.2f}".format(r2_valid)
        acc_valid_ = "{:.2f}".format(acc_valid)
        r2_test_ = "{:.2f}".format(r2_test)
        acc_test_ = "{:.2f}".format(acc_test)
        plt.title('Selected epoch: ' + selected_epoch + ',  val r2:' + r2_valid_ + \
                  ',  val acc:' + acc_valid_ + ',  test r2:' + r2_test_ + ',  test acc:' + acc_test_)
        plt.legend(['pred_val', 'actual_val'], loc='upper left')
        plt.subplot(313)
        xxx_axis = list(range(1, len(preds_tdates) + 1))
        plt.xticks(xxx_axis, preds_tdates, rotation=70)
        plt.stem(xxx_axis, preds, '.', 'x')
        plt.stem(xxx_axis, preds_ground, '.', 'o')
        plt.legend(['pred', 'actual'], loc='upper left')
        img_id = iteration_id + '.png'
        if not os.path.isdir(savepath_imgs):
            os.mkdir(savepath_imgs)
        plt.savefig(savepath_imgs+'\\'+img_id)
        plt.show()
        plt.close()

        # write metrics (R2, accuracy) to CSV
        if not os.path.isfile(savepath_metrics):
            with open(savepath_metrics, 'w', newline='') as csvfile:
                metric_writer = csv.writer(csvfile, delimiter=',')
                metric_writer.writerow(['iteration name','period length','validation r2',
                                        'test r2', 'validation acc', 'test acc'])
                metric_writer.writerow([iteration_id, len(preds), r2_valid,
                                        r2_test, acc_valid, acc_test])
        else:
            with open(savepath_metrics, 'a', newline='') as csvfile:
                metric_writer = csv.writer(csvfile, delimiter=',')
                metric_writer.writerow([iteration_id, len(preds), r2_valid,
                                        r2_test, acc_valid, acc_test])

    # write model configuration once for the loop------------------------------------------------
    with open(savepath_config, 'wb') as f:
        pickle.dump(model_selected.get_config(), f, pickle.HIGHEST_PROTOCOL)

    # load saved model configuration if needed----------------------------------------------------
    with open(savepath_config, 'rb') as f:
        saved_model_configuration = pickle.load(f)

    del history
