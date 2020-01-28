# %% LOOPS THROUGH PARAMETERS WITH 1 DATASET

# %% import base packages

import os
import tensorflow as tf
# import numpy as np
import math
# import operator
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

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers, regularizers, losses
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from import_data import ImportCSVdata_processed
# from import_data import Import3Ddata
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
# path_CSV = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV' # 8700k
# #path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro/Quant Project\Outsourcing/Data Release\mat files 8 multiple all dim'
# path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files 8 multiple 2 dim'
# #path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files temp'
#
# checkpoint_dir = r'C:\Users\jp\TF_checkpoints_WF' # checkpoint weights
# logdir_rootroot = r'C:\Users\jp\TF_logs\WF_1optimizer_' # for tensorboard
# saveDIR_root = r'C:\Users\jp\Google Drive\JP data on ubuntu01\WFpredictions_Final\1optimizer' # for CSV

# Windows on Unbuntu01-----------------------------------------------------------------------------------
path_CSV = r'C:\Users\jp\Google Drive\Data Release\CSV'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple all dim'
path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple 2 dim'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files temp'

checkpoint_dir = r'F:\TF_checkpoints_WF' # checkpoint weights
logdir_rootroot = r'C:\Users\jp\TF_logs\WF_1optimizer_' # for tensorboard
saveDIR_root = r'C:\Users\jp\Google Drive\JP data on ubuntu01\WFpredictions_Final\1optimizer' # for CSV

# %% import data

# tpName = 'KOSPI_HLmid_mrng1030'
# tpName = 'KOSPI_HLmid_mrng'
# tpName = 'KOSPI_PTTP_mrng1030'
# tpName = 'KOSPI_PTTP_mrng'
# tpName = 'KOSPI_HIGH_mrng1030'
tpName = 'KOSPI_LOW_mrng1030'

# varname = ''                      # for importing processed, clipped 3D data
varname = 'processed_'            # for importing processed, clipped CSV data
# varname = 'unprocessed_'          # for importing unprocessed CSV data
# varname = 'processed_unclipped_'    # for importing processed, unclipped CSV data
# varname = 'processed_clipped25'   # for importing processed CSV data, clipped at 25

TestSize = 0
MinBatchMultiple = 3 # mutiple of batchSize_multiple (usually 8);
ValidBatchMultiple = 0 # multiple of MinBatchMultiple; use default if []; for WF, set to 0

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

# %% setup callbacks (the parameters of which do not depend on runs)

checkpoint_modifier = '\cp-{epoch:04d}'
checkpoint_filesuffix = '.hdf5'
checkpoint_path = checkpoint_dir + checkpoint_modifier + checkpoint_filesuffix
cb_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, period=1, verbose=0)

# %% define params

abridged_validationSize = 15 # days to validate historical models for selection of current model

# early_stop_metric = "R2 rolling mean"
mov_windowsSize = 7  # applicable only if rolling mean is used
early_stop_metric = "single metric"

train_optimizer_1st = optimizers.Nadam(clipnorm=1.)
# train_optimizer_2nd = optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=True)
train_activation = tf.nn.elu
train_lossfunc = losses.LogCosh()

LSTMsize = [60, 30]
FCsize = [128]

train_LSTMinputDroprate = 0.5
train_LSTMrecDroprate = 0.0
train_droprate = 0.5
train_l2reg = 0.0001

MinBatchSize = MinBatchMultiple * batchSize_multiple
num_epoch = 2000
min_epochs_ = 10
patience_ = 500
output_unit = 1

interval = 5
start = int(MinBatchSize * (14))
print("First test date is: "+tdates["tradedates_train"][start+1])
input("Press Enter to continue...")
wf_validation_size = 1 # multiple of mini batch size
wfpreds = pd.DataFrame()

Xdim = X_train.shape
minibatch_shape= [MinBatchSize, Xdim[1], Xdim[2]]

log_identifier = 'LSTMsize_'+'_'.join(map(str,LSTMsize))+'_'
logdir_root=logdir_rootroot+log_identifier+time_stamp

if not os.path.exists(saveDIR_root):
    os.mkdir(saveDIR_root)

savepath_preds_csv = saveDIR_root + '\\'+ tpName + '_' + time_stamp + '_preds.csv' # overwrite predictions on same file
savepath_config = saveDIR_root + '\\'+ tpName + '_' + time_stamp + '_config.pkl' # path for model configuration
savepath_models = saveDIR_root + '\\'+ tpName+'_' + time_stamp+'_models'
savepath_imgs = saveDIR_root + '\\'+ tpName+'_' + time_stamp+'_images'
savepath_metrics = saveDIR_root + '\\'+ tpName+'_' + time_stamp+'_metric.csv'

# %% run in loop

print('Run: tensorboard --logdir now:'+logdir_root)
input("Press Enter to continue...")

for i in range(start, Y_train.shape[0]-interval, interval):

    # refresh checkpoint folder
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        time.sleep(5)
        os.makedirs(checkpoint_dir)

    # cut off data for current walk----------------------------------------------------
    X_cut = X_train[:i + interval]
    Y_cut = Y_train[:i + interval]
    tdates_cut = tdates["tradedates_train"][:i + interval]
    print('X_cut dimension: ' + str(X_cut.shape))
    print('Y_cut dimension: ' + str(Y_cut.shape))
    print('tdates_cut dimension: ' + str(tdates_cut.shape))
    if (i % MinBatchSize) != 0:
        print('modulo : ' + str(i % MinBatchSize))
        floorINT = math.floor(i / MinBatchSize)
        Xlength_new = floorINT * MinBatchSize
        X_wf = X_cut[Xlength_new * -1 - interval:interval * -1]
        Y_wf = Y_cut[Xlength_new * -1 - interval:interval * -1]
        tdates_wf = tdates_cut[Xlength_new * -1 - interval:interval * -1]
    else:
        X_wf = X_cut[:interval * -1]
        Y_wf = Y_cut[:interval * -1]
        tdates_wf = tdates_cut[:interval * -1]
    print('X_wf dimension: ' + str(X_wf.shape))
    print('Y_wf dimension: ' + str(Y_wf.shape))
    print('tdates_wf dimension: ' + str(tdates_wf.shape))

    # split train, validation,an test datasets--------------------------------------------
    Xdim = X_wf.shape
    print('Feature dimension: ' + str(Xdim))
    validation_idx = Xdim[0] - wf_validation_size * MinBatchSize

    X_train_ = X_wf[:validation_idx]
    X_valid_ = X_wf[validation_idx:]
    Y_train_ = Y_wf[:validation_idx]
    Y_valid_ = Y_wf[validation_idx:]
    tdates_valid = tdates_wf[validation_idx:]
    X_test_ = X_cut[wf_validation_size * MinBatchSize * -1:]
    Y_test_ = Y_cut[wf_validation_size * MinBatchSize * -1:]
    tdates_test = tdates_cut[wf_validation_size * MinBatchSize * -1:]

    print('X_train dimension: ' + str(X_train_.shape))
    print('Y_train dimension: ' + str(Y_train_.shape))
    print('X_valid dimension: ' + str(X_valid_.shape))
    print('Y_valid dimension: ' + str(Y_valid_.shape))
    print('X_test dimension: ' + str(X_test_.shape))
    print('tdates_test dimension: ' + str(tdates_test.shape))
    print('test period begins on: ' + tdates_test[interval * -1])

    # create model----------------------------------------------------------------------------
    model = create_model_LBALBAFDR(minibatch_shape,
                                   LSTMsize, FCsize,
                                   train_optimizer_1st,
                                   activationF=train_activation,
                                   LSTMinputDroprate=train_LSTMinputDroprate,
                                   LSTMrecDroprate=train_LSTMrecDroprate,
                                   droprate=train_droprate,
                                   l2reg=train_l2reg,
                                   lossfunc=train_lossfunc)

    # setup callbacks ------------------------------------------------------------------------
    logdir_run = r'\\' + tdates_test[interval*-1]
    # cb_tensorboard = TensorBoard(log_dir=logdir_root + logdir_run, histogram_freq=0)

    if early_stop_metric == "single metric":
        cb_trainstop = EarlyStopping(monitor='val_r_square', patience=patience_,
                                   mode='max', restore_best_weights=True)
    elif early_stop_metric == "R2 rolling mean":
        cb_trainstop = create_movMetricStop_callback(stop_patience=patience_,
                                               min_epochs=min_epochs_, logdir=logdir_root+logdir_run)
    else:
        raise NameError('early_stop_metric not defined')

    # CBs = [cb_tensorboard, cb_checkpoint, cb_trainstop, cb_PrintDot()]
    CBs = [cb_checkpoint, cb_trainstop, cb_PrintDot()]

    # set up mixed precision training------------------------------------------------------
    # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    # tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
    # tf.keras.mixed_precision.experimental.set_policy('infer')
    # print(tf.keras.mixed_precision.experimental.global_policy())

    # train with designated deivce(s)----------------------------------------------------------
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    # with tf.device("/device:GPU:0"):
    X_train_ = tf.cast(X_train_, tf.float32)
    Y_train_ = tf.cast(Y_train_, tf.float32)
    X_valid_ = tf.cast(X_valid_, tf.float32)
    Y_valid_ = tf.cast(Y_valid_, tf.float32)

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
        model_selected = selectmodel_basedon_movMetric(X_valid_, Y_valid_, mov_windowsSize, history,
                                   checkpoint_dir, checkpoint_filesuffix, model, metric='val_r_square')
    else:
        raise NameError('early_stop_metric not defined')

    # save selected model-----------------------------------------------------------------------
    model_id = 'model'+str(i)+'.hdf5'
    if not os.path.isdir(savepath_models):
        os.mkdir(savepath_models)
    model_selected.save(savepath_models+'\\'+model_id)

    # load all selected models and get R2 on abridged validation dataset-------------------------
    model_list =os.listdir(savepath_models)
    if 'desktop.ini' in model_list:
        model_list.remove('desktop.ini')
    r2_hist = []
    for modelfile in model_list:
        model_temp = create_model_LBALBAFDR(minibatch_shape,
                                       LSTMsize, FCsize,
                                       train_optimizer_1st,
                                       activationF=train_activation,
                                       LSTMinputDroprate=train_LSTMinputDroprate,
                                       LSTMrecDroprate=train_LSTMrecDroprate,
                                       droprate=train_droprate,
                                       l2reg=train_l2reg,
                                       lossfunc=train_lossfunc)
        model_temp.load_weights(savepath_models+'\\'+modelfile)
        pred_temp = model_temp.predict(X_valid_)
        pred_temp = pred_temp[abridged_validationSize * -1:]
        pred_ground = Y_valid_[abridged_validationSize * -1:]
        r2_temp = r2_score(pred_ground, pred_temp)
        r2_hist.append(r2_temp)

    max_tmp = max(r2_hist)
    max_idxs_tmp = [ii for ii, vv in enumerate(r2_hist) if vv == max_tmp]
    max_idx_tmp = max_idxs_tmp[-1]
    model_for_this_walk = create_model_LBALBAFDR(minibatch_shape,
                                       LSTMsize, FCsize,
                                       train_optimizer_1st,
                                       activationF=train_activation,
                                       LSTMinputDroprate=train_LSTMinputDroprate,
                                       LSTMrecDroprate=train_LSTMrecDroprate,
                                       droprate=train_droprate,
                                       l2reg=train_l2reg,
                                       lossfunc=train_lossfunc)
    model_for_this_walk.load_weights(savepath_models+'\\'+model_list[max_idx_tmp])

    # make prediction with selected model on test dataset----------------------------------------
    preds = model_for_this_walk.predict(X_test_)
    preds = preds[interval * -1:]
    preds = preds * sigma + mu
    preds_tdates = tdates_test[interval * -1:]
    preds_ground = Y_test_[interval * -1:]
    preds_ground = preds_ground * sigma + mu
    r2_test = r2_score(preds_ground, preds)

    # plots--------------------------------------------------------------------------------------
    plt.figure
    plt.subplot(211)
    plt.plot(history.history['r_square'])
    plt.plot(history.history['val_r_square'])
    plt.title(preds_tdates[0])
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.subplot(212)
    xx_axis = list(range(1,len(preds_tdates)+1))
    plt.xticks(xx_axis,preds_tdates, rotation=70)
    plt.stem(xx_axis, preds, '.','x')
    plt.stem(xx_axis, preds_ground, '.','o')
    plt.legend(['pred', 'actual'], loc='upper left')
    # plt.show()
    img_id = 'interval'+str(i)+'.png'
    if not os.path.isdir(savepath_imgs):
        os.mkdir(savepath_imgs)
    plt.savefig(savepath_imgs+'\\'+img_id)
    plt.close()

    dataf = {'tradedate': preds_tdates.tolist(), 'prediction': preds.tolist()}
    dataF = pd.DataFrame.from_dict(dataf)
    wfpreds = pd.concat([wfpreds, dataF])

    # save predcition for current walk in CSV; ---------------------------------------------------
    wfpreds.to_csv(savepath_preds_csv, index=False)

    # write metrics (R2) to CSV
    if i == start:
        with open(savepath_metrics, 'w', newline='') as csvfile:
            metric_writer = csv.writer(csvfile, delimiter=',')
            metric_writer.writerow(['period start','period length','validation r2', 'test r2'])
            metric_writer.writerow([preds_tdates[0], len(preds), max_tmp, r2_test])
    else:
        with open(savepath_metrics, 'a', newline='') as csvfile:
            metric_writer = csv.writer(csvfile, delimiter=',')
            metric_writer.writerow([preds_tdates[0], len(preds), max_tmp, r2_test])

    # write model configuration once for the loop------------------------------------------------
    if i == start:
        with open(savepath_config, 'wb') as cf:
            pickle.dump(model_temp.get_config(), cf, pickle.HIGHEST_PROTOCOL)

    # load saved model configuration if needed----------------------------------------------------
    # with open(savepath_config, 'rb') as cf:
    #     saved_model_configuration = pickle.load(cf)

    del model, model_for_this_walk, model_selected, model_temp, history
    del X_cut, Y_cut, tdates_cut, X_wf, Y_wf, tdates_wf, \
        Xdim, validation_idx, X_train_, Y_train_, X_valid_, Y_valid_, tdates_valid, \
        X_test_, Y_test_, tdates_test, logdir_run, cb_trainstop, \
        CBs, model_id, model_list, r2_hist, modelfile, pred_temp, pred_ground, \
        r2_temp, max_tmp, max_idxs_tmp, max_idx_tmp, preds, preds_tdates, \
        preds_ground, r2_test, xx_axis, img_id, dataf, dataF#, cb_tensorboard



# %% save predictions as mat file

csvfile.close()
savefile_mat = saveDIR_root + '\\' + tpName + '_' + time_stamp + '.mat'
spio.savemat(savefile_mat, {'struct': wfpreds.to_dict("list")})