# %% LOOP THROUGH DATASETS WITH 1 SET OF PARAMETERS

# %% import base packages

import packages_utilities
packages_utilities.select_gpu("DESKTOP-DNL8QNB")

import os
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

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers, regularizers, losses
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from import_data import ImportCSVdata_processed, Import3Ddata
from packages_training import r_square, propAcc, create_model_LBALBAFDR, \
    create_movMetricStop_callback, cb_PrintDot, selectmodel_basedon_movMetric
# import sys

print(tf.__version__)
print(tf.keras.__version__)

# %% check GPU & control gpu memory usage -- not necesary for windows 10 (?)

# tf.debugging.set_log_device_placement(True) # shows which device is being used
tf.config.set_soft_device_placement(True) # assigns another device if designated one isn't available

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
path_CSV = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV' # 8700k
#path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro/Quant Project\Outsourcing/Data Release\mat files 8 multiple all dim'
path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files 8 multiple 2 dim'
#path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files temp'

checkpoint_dir = r'C:\Users\jp\TF_checkpoints_Tests' # checkpoint weights
logdir_rootroot = r'C:\Users\jp\TF_logs\Main\4data_' # for tensorboard
saveDIR_root = r'C:\Users\jp\Google Drive\TF_Predictions\Predictions_Tests\4data' # for CSV

# Windows on Unbuntu01-----------------------------------------------------------------------------------
# path_CSV = r'C:\Users\jp\Google Drive\Data Release\CSV'
# # path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple all dim'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple 2 dim'
# # path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files temp'
#
# checkpoint_dir = r'F:\TF_checkpoints_Tests' # checkpoint weights
# logdir_rootroot = r'C:\Users\jp\TF_logs\Main\4data_' # for tensorboard
# saveDIR_root = r'C:\Users\jp\Google Drive\TF_Predictions\Predictions_Tests\4data' # for CSV

# %% import data

tpName = 'KOSPI_HLmid_mrng1030'
# tpName = 'KOSPI_HLmid_mrng'
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

    # from import_data import ImportCSVdata_unprocessed

    data_dict[i] = imported_

# %% define params -- using nested lists to recycle code from main_1daya.py

# abridged_validationSize = 15 # days to validate historical models for selection of current model

# early_stop_metric = "R2 rolling mean"
mov_windowsSize = 7  # applicable only if rolling mean is used
early_stop_metric = "single metric"

train_optimizer = [
#     [optimizers.Adadelta(clipnorm=1.),'_vanila']
#     [optimizers.SGD(clipnorm=1.),'_vanila']
#     [optimizers.SGD(clipnorm=1., momentum=0., nesterov=True), '_Nstrv']
#     [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=False), '_m05_Nstrv']
#     [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=True), '_m05']
    [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=True), '_m09_Nstrv']
    # [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=False), '_m09']
#     [optimizers.RMSprop(clipnorm=1.),'_vanila']
#     [optimizers.Adagrad(clipnorm=1.),'_vanila']
#     [optimizers.Adamax(clipnorm=1.),'_vanila']
    # [optimizers.Nadam(clipnorm=1.),'_vanila']
#     [optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False, clipnorm=1.),'_vanila']
]
train_activation = [
    # tf.nn.leaky_relu
    # tf.nn.relu
    # tf.nn.relu6
    # tf.nn.selu
    tf.nn.elu
]

train_lossfunc = [
#     losses.MeanSquaredError()
#     losses.MeanAbsoluteError()
    losses.Huber(delta=1.0)
#     losses.LogCosh()
]

LSTMsize = [
    # [720, 720]
    # [720, 480]
    # [480, 480]
    # [480, 240]
    # [240, 240]
    [240, 120]
    # [120, 120]
    # [120, 60]
    # [60, 60]
    # [60, 30]
    # [30, 30]
]

FCsize = [
    [128]
]

train_LSTMinputDroprate = 0.5
train_LSTMrecDroprate = 0.0
train_droprate = 0.5
train_l2reg = 0.0001

num_epoch = 2000
min_epochs_ = 50
patience_= 1000
output_unit = 1

# log_identifier = 'LSTMsize_'+'_'.join(map(str,LSTMsize[0]))+'_'
log_identifier = ''
logdir_root=logdir_rootroot+log_identifier+time_stamp

if not os.path.exists(saveDIR_root):
    os.mkdir(saveDIR_root)

savepath_preds_csv = saveDIR_root + '\\'+ tpName + '_' + time_stamp + '_preds.csv' # overwrite predictions on same file
savepath_config = saveDIR_root + '\\'+ tpName + '_' + time_stamp + '_config.pkl' # path for model configuration
savepath_models = saveDIR_root + '\\'+ tpName+'_' + time_stamp+'_models'
savepath_imgs = saveDIR_root + '\\'+ tpName+'_' + time_stamp+'_images'
savepath_metrics = saveDIR_root + '\\'+ tpName+'_' + time_stamp+'_metric.csv'

# %% setup callbacks (the parameters of which do not depend on runs)

checkpoint_modifier = '\cp-{epoch:04d}'
checkpoint_filesuffix = '.hdf5'
checkpoint_path = checkpoint_dir + checkpoint_modifier + checkpoint_filesuffix
if early_stop_metric == "single metric":
    bestonly = True
else:
    bestonly = False

# %% run in loop

train_optimizer_ = train_optimizer[0][0]
optimizer_name = train_optimizer_.__dict__.get("_name") + train_optimizer[0][1]

train_activation_ = train_activation[0]
activation_name = train_activation_.__name__

train_lossfunc_ = train_lossfunc[0]
lossfunc_name = train_lossfunc_.__dict__.get("name")

LSTMsize_ = LSTMsize[0]
LSTMname = 'LSTMsize_' + '_'.join(map(str, LSTMsize_))

FCsize_ = FCsize[0]
FCname = 'FCsize_' + '_'.join(map(str, FCsize_))

run_id = optimizer_name+'_'+activation_name+'_'+lossfunc_name+'_'+LSTMname+'_'+FCname
logdir_root=logdir_root+'_'+run_id
print('Run: tensorboard --logdir now:'+logdir_root)
input("Press Enter to continue...")

for i in range(len(data_dict)):

    # refresh checkpoint folder
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        time.sleep(5)
        os.makedirs(checkpoint_dir)    

    print("data #: "+str(i))
    X_train = data_dict[i]["X_train"]
    Y_train = data_dict[i]["Y_train"]
    X_valid = data_dict[i]["X_valid"]
    Y_valid = data_dict[i]["Y_valid"]
    X_test = data_dict[i]["X_test"]
    Y_test = data_dict[i]["Y_test"]
    batchSize_multiple = data_dict[i]["batchSize_multiple"]
    tdates_train = data_dict[i]['tradedates']["tradedates_train"]
    tdates_valid = data_dict[i]['tradedates']["tradedates_valid"]
    tdates_test = data_dict[i]['tradedates']["tradedates_test"]
    mu = imported_["mu"]
    sigma = imported_["sigma"]

    print('X_train dimension: ' + str(X_train.shape))
    print('Y_train dimension: ' + str(Y_train.shape))
    print('X_valid dimension: ' + str(X_valid.shape))
    print('Y_valid dimension: ' + str(Y_valid.shape))
    print('X_test dimension: ' + str(X_test.shape))
    print('tdates_test dimension: ' + str(tdates_test.shape))
    print('tdates_test dimension: ' + str(tdates_test.shape))

    MinBatchSize = MinBatchMultiple * batchSize_multiple
    Xdim = X_train.shape
    minibatch_shape= [MinBatchSize, Xdim[1], Xdim[2]]

    # setup callbacks ------------------------------------------------------------------------
    logdir_run=r'\\data'+str(i)
    cb_tensorboard = TensorBoard(log_dir=logdir_root + logdir_run, histogram_freq=0)

    if early_stop_metric == "single metric":
        cb_trainstop = EarlyStopping(monitor='val_r_square', patience=patience_,
                                mode='max', restore_best_weights=True)
    elif early_stop_metric == "R2 rolling mean":
        cb_trainstop = create_movMetricStop_callback(stop_patience=patience_,
                                            min_epochs=min_epochs_, logdir=logdir_root+logdir_run)
    else:
        raise NameError('early_stop_metric not defined')

    cb_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=bestonly, save_weights_only=True, period=1,
                                    verbose=0)

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
    # X_train = tf.cast(X_train, tf.float32)
    # Y_train = tf.cast(Y_train, tf.float32)
    # X_valid = tf.cast(X_valid, tf.float32)
    # Y_valid = tf.cast(Y_valid, tf.float32)

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

    print(run_id)
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
        model_selected = model
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
    preds_ground_valid = Y_valid
    preds_ground_valid = preds_ground_valid * sigma + mu
    preds_ground_valid = preds_ground_valid.reshape([len(preds_ground_valid), 1])
    r2_valid = r2_score(preds_ground_valid, preds_valid)

    acc_valid = accuracy_score(np.sign(preds_ground_valid), np.sign(preds_valid))

    # make prediction with trained model--------------------------------------------------------------
    # if not tf.equal(tf.size(X_test),0):
    if X_test.any():
        preds = model_selected.predict(X_test)
        preds = preds * sigma + mu
        preds = preds.reshape([len(preds), 1])
        preds_tdates = tdates_test
        preds_ground = Y_test
        preds_ground = preds_ground * sigma + mu
        preds_ground = preds_ground.reshape([len(preds_ground), 1])
        r2_test = r2_score(preds_ground, preds)

        acc_test = accuracy_score(np.sign(preds_ground), np.sign(preds))

        # plots--------------------------------------------------------------------------------------
        plt.figure(figsize=[8.27,11.69])
        plt.subplot(311)
        plt.plot(history.history['r_square'])
        plt.plot(history.history['val_r_square'])
        plt.title(optimizer_name + '_' + activation_name + '_' + lossfunc_name + '_' + LSTMname + '_' + FCname + ' ' +
                  preds_tdates[0])
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
        img_id = 'interval' + str(i) + '.png'
        if not os.path.isdir(savepath_imgs):
            os.mkdir(savepath_imgs)
        plt.savefig(savepath_imgs + '\\' + img_id)
        plt.show()
        plt.close()

    del model, model_selected, history
