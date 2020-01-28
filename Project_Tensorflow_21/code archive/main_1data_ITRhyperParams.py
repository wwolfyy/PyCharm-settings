# %% LOOPS THROUGH PARAMETERS WITH 1 DATASET

# %% import base packages

import os
import tensorflow as tf
import numpy as np
import math
import operator
import datetime
import csv
import pandas as pd
from sklearn.metrics import r2_score
from scipy import io as spio
import shutil
import pickle

import time
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers, regularizers, losses
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from import_data import ImportCSVdata_processed
from import_data import Import3Ddata
from packages_training import r_square, propAcc, create_model_LBALBAFDR, \
    create_movMetricStop_callback, cb_PrintDot, selectmodel_basedon_movMetric
import sys

print(tf.__version__)
print(tf.keras.__version__)

# %% check GPU & control gpu memory usage -- not necesary for windows 10 (?)

# tf.debugging.set_log_device_placement(True) # shows which device is being used
tf.config.set_soft_device_placement(True)  # assigns another device if designated one isn't available

device_lib.list_local_devices()
if not tf.config.gpu.get_per_process_memory_growth():
    tf.config.gpu.set_per_process_memory_fraction(0.7)
    tf.config.gpu.set_per_process_memory_growth(True)

print(tf.config.gpu.get_per_process_memory_fraction())
print(tf.config.gpu.get_per_process_memory_growth())

# %% set file paths

time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Linux--------------------------------------------------------------------------------------------------
# path_CSV = '/home/lstm/Desktop/GoogleDrive_local/Data Release/CSV'
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files 8 multiple 2 dim' # Linux
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files 8 multiple all dim'
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files temp'


# 8700k--------------------------------------------------------------------------------------------------
path_CSV = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV'  # 8700k
# path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro/Quant Project\Outsourcing/Data Release\mat files 8 multiple all dim'
path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files 8 multiple 2 dim'
# path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files temp'

checkpoint_dir = r'C:\Users\jp\TF_checkpoints_WIP'  # checkpoint weights
logdir_rootroot = r'C:\Users\jp\TF_logs\1data_'  # for tensorboard
saveDIR_root = r'C:\Users\jp\Google Drive\JP data on ubuntu01\WFpredictions_WIP\1data'  # for CSV

# Windows on Unbuntu01-----------------------------------------------------------------------------------
# path_CSV = r'C:\Users\jp\Google Drive\Data Release\CSV'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple all dim'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple 2 dim'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files temp'

# checkpoint_dir = r'C:\Users\jp\TF_checkpoints_WIP' # checkpoint weights
# logdir_rootroot = r'C:\Users\jp\TF_logs\1data_' # for tensorboard
# saveDIR_root = r'C:\Users\jp\Google Drive\JP data on ubuntu01\WFpredictions_Final\1data' # for CSV

# %% import data

tpName = 'KOSPI_HLmid_mrng1030'
# tpName = 'KOSPI_HLmid_mrng'
# tpName = 'KOSPI_PTTP_mrng1030'
# tpName = 'KOSPI_PTTP_mrng'
# tpName = 'KOSPI_HIGH_mrng1030'
# tpName = 'KOSPI_LOW_mrng1030'

# varname = ''                      # for importing processed, clipped 3D data
varname = 'processed_'  # for importing processed, clipped CSV data
# varname = 'unprocessed_'          # for importing unprocessed CSV data
# varname = 'processed_unclipped_'    # for importing processed, unclipped CSV data
# varname = 'processed_clipped25'   # for importing processed CSV data, clipped at 25

TestSize = 0
MinBatchMultiple = 3  # mutiple of batchSize_multiple (usually 8);
ValidBatchMultiple = 3  # multiple of MinBatchMultiple; use default if []; for WF, set to 0

# from import_data import Import3Ddata
# X_train, Y_train, X_valid, Y_valid, X_test, Y_test, batchSize_multiple, tradedates, mu, sigma = \
#     Import3Ddata(tpName, platform, MinBatchMultiple, ValidBatchMultiple, TestSize)
# imported_ = Import3Ddata(tpName, platform, MinBatchMultiple, ValidBatchMultiple, TestSize)

imported_ = ImportCSVdata_processed(path_CSV, path_mat, tpName, varname,
                                    MinBatchMultiple, ValidBatchMultiple, TestSize)

# from import_data import ImportCSVdata_unprocessed

X_train = imported_["X_train"]
Y_train = imported_["Y_train"]
X_valid = imported_["X_valid"]
Y_valid = imported_["Y_valid"]
X_test = imported_["X_test"]
Y_test = imported_["Y_test"]
batchSize_multiple = imported_["batchSize_multiple"]
tdates = imported_["tradedates"]
mu = imported_["mu"]
sigma = imported_["sigma"]

print('X_train dimension: ' + str(X_train.shape))
print('Y_train dimension: ' + str(Y_train.shape))
print('X_valid dimension: ' + str(X_valid.shape))
print('Y_valid dimension: ' + str(Y_valid.shape))
print('X_test dimension: ' + str(X_test.shape))
print('tdates_test dimension: ' + str(tdates["tradedates_test"].shape))

# %% setup callbacks (the parameters of which do not depend on runs)

checkpoint_modifier = '\cp-{epoch:04d}'
checkpoint_filesuffix = '.hdf5'
checkpoint_path = checkpoint_dir + checkpoint_modifier + checkpoint_filesuffix
cb_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, period=1, verbose=0)

# %% define params

# abridged_validationSize = 15 # days to validate historical models for selection of current model

# early_stop_metric = "R2 rolling mean"
mov_windowsSize = 7  # applicable only if rolling mean is used
early_stop_metric = "single metric"

train_optimizer = [
        [optimizers.Adadelta(clipnorm=1.),'_Adadelta'],
        [optimizers.SGD(clipnorm=1.),'_vanila'],
        [optimizers.SGD(clipnorm=1., momentum=0., nesterov=True), '_Nstrv'],
        [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=False), '_m05_Nstrv'],
        [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=True), '_m05'],
        [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=True), '_m09_Nstrv'],
        [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=False), '_m09'],
        [optimizers.RMSprop(clipnorm=1.),'_RMSprop'],
        [optimizers.Adagrad(clipnorm=1.),'_Adagrad'],
        [optimizers.Adamax(clipnorm=1.),'Adamax'],
        [optimizers.Nadam(clipnorm=1.),'_vanila'],
        [optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False, clipnorm=1.),'Adam']
]
train_activation = [
    # tf.nn.leaky_relu,
    # tf.nn.relu,
    # tf.nn.relu6,
    # tf.nn.selu,
    tf.nn.elu
]
train_lossfunc = [
    # losses.MeanSquaredError(),
    # losses.MeanAbsoluteError(),
    # losses.Huber(delta=1.0),
    losses.LogCosh()
]

LSTMsize = [
    # [720, 720],
    # [720, 480],
    # [480, 480],
    # [480, 240],
    # [240, 240],
    [240, 120]
    # [120, 120],
    # [120, 60],
    # [60, 60],
    # [60, 30],
    # [30, 30]
]

FCsize = [
    [128]
]

train_LSTMinputDroprate = 0.5
train_LSTMrecDroprate = 0.0
train_droprate = 0.5
train_l2reg = 0.0001

MinBatchSize = MinBatchMultiple * batchSize_multiple
num_epoch = 2000
min_epochs_ = 50
patience_ = 1000
output_unit = 1

Xdim = X_train.shape
minibatch_shape = [MinBatchSize, Xdim[1], Xdim[2]]

log_identifier = 'LSTMsize_' + '_'.join(map(str, LSTMsize[0])) + '_'
logdir_root = logdir_rootroot + log_identifier + time_stamp

if not os.path.exists(saveDIR_root):
    os.mkdir(saveDIR_root)

savepath_preds_csv = saveDIR_root + '\\' + tpName + '_' + time_stamp + '_preds.csv'  # overwrite predictions on same file
savepath_config = saveDIR_root + '\\' + tpName + '_' + time_stamp + '_config.pkl'  # path for model configuration
savepath_models = saveDIR_root + '\\' + tpName + '_' + time_stamp + '_models'
savepath_imgs = saveDIR_root + '\\' + tpName + '_' + time_stamp + '_images'
savepath_metrics = saveDIR_root + '\\' + tpName + '_' + time_stamp + '_metric.csv'

# %% run in loop

print('Run: tensorboard --logdir now:' + logdir_root)
input("Press Enter to continue...")

for i in range(len(train_optimizer)):

    # refresh checkpoint folder
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        time.sleep(5)
        os.makedirs(checkpoint_dir)

    train_optimizer_ = train_optimizer[i][0]
    optimizer_name = train_optimizer_.__dict__.get("_name") + train_optimizer[i][1]
    for train_activation_ in train_activation:
        activation_name = train_activation_.__name__

        for train_lossfunc_ in train_lossfunc:
            lossfunc_name = train_lossfunc_.__dict__.get("name")

            for LSTMsize_ in LSTMsize:
                LSTMname = 'LSTMsize_' + '_'.join(map(str, LSTMsize_))

                for FCsize_ in FCsize:
                    FCname = 'FCsize_' + '_'.join(map(str, FCsize_))

                    iteratin_id = optimizer_name + '_' + activation_name + '_' + lossfunc_name + '_' + LSTMname + '_' + FCname

                    # setup callbacks ------------------------------------------------------------------------
                    logdir_run = r'\\' + iteratin_id
                    cb_tensorboard = TensorBoard(log_dir=logdir_root + logdir_run, histogram_freq=0)

                    if early_stop_metric == "single metric":
                        cb_trainstop = EarlyStopping(monitor='val_r_square', patience=patience_,
                                                     mode='max', restore_best_weights=True)
                    elif early_stop_metric == "R2 rolling mean":
                        cb_trainstop = create_movMetricStop_callback(stop_patience=patience_,
                                                                     min_epochs=min_epochs_,
                                                                     logdir=logdir_root + logdir_run)
                    else:
                        raise NameError('early_stop_metric not defined')

                    CBs = [cb_tensorboard, cb_checkpoint, cb_trainstop, cb_PrintDot()]

                    # set up mixed precision training------------------------------------------------------
                    # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
                    # tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
                    # tf.keras.mixed_precision.experimental.set_policy('infer')
                    # print(tf.keras.mixed_precision.experimental.global_policy())

                    # train with designated deivce(s)----------------------------------------------------------
                    # strategy = tf.distribute.MirroredStrategy()
                    # with strategy.scope():
                    # with tf.device("/device:GPU:0"):
                    X_train_ = tf.cast(X_train, tf.float32)
                    Y_train_ = tf.cast(Y_train, tf.float32)
                    X_valid_ = tf.cast(X_valid, tf.float32)
                    Y_valid_ = tf.cast(Y_valid, tf.float32)

                    # create & run model----------------------------------------------------------------------------
                    model = create_model_LBALBAFDR(minibatch_shape,
                                                   LSTMsize_, FCsize_,
                                                   train_optimizer_,
                                                   activationF=train_activation_,
                                                   LSTMinputDroprate=train_LSTMinputDroprate,
                                                   LSTMrecDroprate=train_LSTMrecDroprate,
                                                   droprate=train_droprate,
                                                   l2reg=train_l2reg,
                                                   lossfunc=train_lossfunc_)

                    model.fit(X_train_, Y_train_,
                              validation_data=(X_valid_, Y_valid_),
                              batch_size=MinBatchSize,
                              epochs=num_epoch,
                              verbose=2,
                              callbacks=CBs,
                              shuffle=False
                              )

                    history = model.fit(X_train_, Y_train_,
                                        validation_data=(X_valid_, Y_valid_),
                                        batch_size=MinBatchSize,
                                        epochs=num_epoch,
                                        verbose=0,
                                        callbacks=CBs,
                                        shuffle=False
                                        )

                    # select model from checkpoints------------------------------------------------------------
                    if early_stop_metric == "single metric":
                        model_selected = model
                    elif early_stop_metric == "R2 rolling mean":
                        model_selected = selectmodel_bas + '_' + edon_movMetric(X_valid_, Y_valid_, mov_windowsSize,
                                                                                history,
                                                                                checkpoint_dir, checkpoint_filesuffix,
                                                                                model, metric='val_r_square')
                    else:
                        raise NameError('early_stop_metric not defined')

                    # make prediction with selected model on validation dataset----------------------------------------
                    preds_valid = model_selected.predict(X_valid)
                    preds_valid = preds_valid * sigma + mu
                    preds_tdates_valid = tdates["tradedates_valid"]
                    preds_ground_valid = Y_valid
                    preds_ground_valid = preds_ground_valid * sigma + mu
                    r2_valid = r2_score(preds_ground_valid, preds_valid)

                    # make prediction with selected model on test dataset----------------------------------------
                    preds = model_selected.predict(X_test)
                    preds = preds[TestSize * -1:]
                    preds = preds * sigma + mu
                    preds_tdates = tdates["tradedates_test"]
                    preds_ground = Y_test
                    preds_ground = preds_ground * sigma + mu
                    r2_test = r2_score(preds_ground, preds)

                    # plots--------------------------------------------------------------------------------------
                    plt.figure
                    plt.subplot(211)
                    plt.plot(history.history['r_square'])
                    plt.plot(history.history['val_r_square'])
                    plt.title(optimizer_name + '_' + activation_name + '_' + lossfunc_name)
                    plt.ylabel('R^2')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'valid'], loc='upper left')
                    plt.subplot(212)
                    xx_axis = list(range(1, len(preds_tdates) + 1))
                    plt.xticks(xx_axis, preds_tdates, rotation=70)
                    plt.stem(xx_axis, preds, '.', 'x')
                    plt.stem(xx_axis, preds_ground, '.', 'o')
                    plt.legend(['pred', 'actual'], loc='upper left')
                    # plt.show()
                    img_id = 'interval' + str(i) + '.png'
                    if not os.path.isdir(savepath_imgs):
                        os.mkdir(savepath_imgs)
                    plt.savefig(savepath_imgs + '\\' + img_id)
                    plt.close()

                    # write metrics (R2) to CSV
                    if i == 0:
                        with open(savepath_metrics, 'w', newline='') as csvfile:
                            metric_writer = csv.writer(csvfile, delimiter=',')
                            metric_writer.writerow(['iteration name', 'period length', 'validation r2', 'test r2'])
                            metric_writer.writerow([iteratin_id, len(preds), r2_valid, r2_test])
                    else:
                        with open(savepath_metrics, 'a', newline='') as csvfile:
                            metric_writer = csv.writer(csvfile, delimiter=',')
                            metric_writer.writerow([iteratin_id, len(preds), r2_valid, r2_test])

                    # write model configuration once for the loop------------------------------------------------
                    if i == 0:
                        with open(savepath_config, 'wb') as f:
                            pickle.dump(model_selected.get_config(), f, pickle.HIGHEST_PROTOCOL)

                    # load saved model configuration if needed----------------------------------------------------
                    with open(savepath_config, 'rb') as f:
                        saved_model_configuration = pickle.load(f)

                    del model, model_selected, history