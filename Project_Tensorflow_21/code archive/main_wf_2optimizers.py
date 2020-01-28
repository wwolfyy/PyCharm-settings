# %% LOOPS THROUGH PARAMETERS WITH 1 DATASET
# Memory (CPU) usage racks up with iteration as more graphs are added on each iteration
# DO NOT USE UNTIL SOLUTION IS AVAILABLE TO FLUSH MEMORY AFTER EACH ITERATION

# %% import base packages

import packages_utilities
packages_utilities.select_gpu("DESKTOP-DNL8QNB")

import os
pc_name = os.environ['COMPUTERNAME']

import os
import tensorflow as tf
import numpy as np
import math
# import operator
import datetime
import csv
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
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
if pc_name == "DESKTOP-DNL8QNB":
    path_CSV = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV' # 8700k
    #path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro/Quant Project\Outsourcing/Data Release\mat files 8 multiple all dim'
    path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files 8 multiple 2 dim'
    #path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files temp'

    checkpoint_dir = r'C:\Users\jp\TF_checkpoints_WF'  # checkpoint weights
    logdir_rootroot = r'C:\Users\jp\Google Drive\TF_logs\Logs_WF\\'  # for tensorboard
    saveDIR_root = r'C:\Users\jp\Google Drive\TF_Predictions\WFpredictions_Final'  # for CSV

if pc_name == "DESKTOP-U9PDCSR":
    path_CSV = r'C:\Users\jp\Google Drive\Data Release\CSV'
    # path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple all dim'
    path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple 2 dim'
    # path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files temp'

    checkpoint_dir = r'F:\TF_checkpoints_WF' # checkpoint weights
    logdir_rootroot = r'C:\Users\jp\Google Drive\TF_logs\Logs_WF\\' # for tensorboard
    saveDIR_root = r'C:\Users\jp\Google Drive\TF_Predictions\WFpredictions_Final'  # for CSV

# %% import data

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

# %% define params

abridged_validationSize = 15 # days to validate historical models for selection of current model
re_intialize_intervals = 5 # after this number of intervals, initial weights for 1st model will be re-initialized

# early_stop_metric = "R2 rolling mean"
mov_windowsSize = 7  # applicable only if rolling mean is used
early_stop_metric = "single metric"

train_optimizer_1st = optimizers.Nadam(lr=0.002, clipnorm=1.)
# train_optimizer_1st = optimizers.SGD(lr=0.1, clipnorm=1., momentum=0.9, nesterov=True)
train_optimizer_1st = optimizers.SGD(lr=0.01, clipnorm=1., momentum=0.9, nesterov=True)

train_activation = tf.nn.elu

# train_lossfunc = losses.MeanSquaredError()
train_lossfunc = losses.MeanAbsoluteError()
# train_lossfunc = losses.LogCosh()
# train_lossfunc = losses.Huber(delta=0.3)

LSTMsize = [240, 120]

FCsize = [128]

train_LSTMinputDroprate = 0.5
train_LSTMrecDroprate = 0.0
train_droprate = 0.5
train_l2reg = 0.0001

MinBatchSize = MinBatchMultiple * batchSize_multiple
num_epoch = 2000
num_epoch_2nd = 500
min_epochs_ = 10
patience_ = 500
patience_2nd = 500
output_unit = 1

interval = 5
start = int(MinBatchSize * (14))
firstdate=tdates["tradedates_train"][start]
wf_validation_size = 1 # multiple of mini batch size
wfpreds = pd.DataFrame()

Xdim = X_train.shape
minibatch_shape= [MinBatchSize, Xdim[1], Xdim[2]]

run_identifier = '2optimizers'
logdir_root=logdir_rootroot+'_'+tpName+'_'+run_identifier+'_'+time_stamp

if not os.path.exists(saveDIR_root):
    os.mkdir(saveDIR_root)

savepath_preds_csv = saveDIR_root + '\\'+ tpName+'_'+run_identifier + '_' + time_stamp + '_preds.csv' # overwrite predictions on same file
savepath_config1 = saveDIR_root + '\\'+ tpName+'_'+run_identifier + '_' + time_stamp + '_config_model1.pkl' # path for model configuration
savepath_config2 = saveDIR_root + '\\'+ tpName+'_'+run_identifier + '_' + time_stamp + '_config_model2.pkl' # path for model configuration
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

# %% run in loop

print('Run: tensorboard --logdir "'+logdir_root+'"')
print("\nRunning walk-forward modeling & prediciton with following parameters:")
print('TP : '+tpName)
print('optimizer #1: '+train_optimizer_1st.__dict__['_name'])
print('learning rate: '+str(train_optimizer_1st.__dict__['_hyper']['learning_rate']))
print('optimizer #2: '+train_optimizer_2nd.__dict__['_name'])
print('learning rate: '+str(train_optimizer_2nd.__dict__['_hyper']['learning_rate']))
print('activation: '+train_activation.__name__)
print('loss functions: '+train_lossfunc.__dict__['name'])
if train_lossfunc.__dict__['name'] == 'huber_loss':
    print('delta: '+str(train_lossfunc.get_config()['delta']))
print('LSTM structures: '+'_'.join(map(str, LSTMsize)))
print('FC structures: '+'_'.join(map(str, FCsize)))
print('final validation size: '+str(abridged_validationSize))
print('early stop metric: '+early_stop_metric)
print('num_epoch: '+str(num_epoch))
print('early stop patience: '+str(patience_))
print("first test date : "+firstdate)
# print('save path: '+saveDIR_root)
input('\nContinue?')

# create 1st model for weight initialization
model_ = create_model_LBALBAFDR(minibatch_shape,
                                LSTMsize, FCsize,
                                train_optimizer_2nd,
                                activationF=train_activation,
                                LSTMinputDroprate=train_LSTMinputDroprate,
                                LSTMrecDroprate=train_LSTMrecDroprate,
                                droprate=train_droprate,
                                l2reg=train_l2reg,
                                lossfunc=train_lossfunc)
saved_random_weights = model_.get_weights()

# crete 2nd model for acutal training
model = create_model_LBALBAFDR(minibatch_shape,
                               LSTMsize, FCsize,
                               train_optimizer_1st,
                               activationF=train_activation,
                               LSTMinputDroprate=train_LSTMinputDroprate,
                               LSTMrecDroprate=train_LSTMrecDroprate,
                               droprate=train_droprate,
                               l2reg=train_l2reg,
                               lossfunc=train_lossfunc)
model_temp = model
model_for_this_walk = model

for i in range(start, Y_train.shape[0]-interval, interval):

    # refresh checkpoint folder
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        time.sleep(3)
        while not os.path.isdir(checkpoint_dir):
            print('Creating ' + checkpoint_dir)
            time.sleep(3)
            os.makedirs(checkpoint_dir)
        print(checkpoint_dir + ' created')

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

    cb_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=bestonly, save_weights_only=True, period=1,
                                    verbose=0)        

    # CBs = [cb_tensorboard, cb_checkpoint, cb_trainstop, cb_PrintDot()]
    CBs = [cb_checkpoint, cb_trainstop, cb_PrintDot()]

    # set up mixed precision training------------------------------------------------------
    # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    # tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
    # tf.keras.mixed_precision.experimental.set_policy('infer')
    # print(tf.keras.mixed_precision.experimental.global_policy())
    X_train_ = tf.cast(X_train_, tf.float32)
    Y_train_ = tf.cast(Y_train_, tf.float32)
    X_valid_ = tf.cast(X_valid_, tf.float32)
    Y_valid_ = tf.cast(Y_valid_, tf.float32)

    # train with designated deivce(s)----------------------------------------------------------
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

    # with tf.device("/device:GPU:0"):

    # re-initialize weights for 1st model periodically
    if (i-start) % (interval*re_intialize_intervals) == 0:
        model_ = create_model_LBALBAFDR(minibatch_shape,
                                LSTMsize, FCsize,
                                train_optimizer_2nd,
                                activationF=train_activation,
                                LSTMinputDroprate=train_LSTMinputDroprate,
                                LSTMrecDroprate=train_LSTMrecDroprate,
                                droprate=train_droprate,
                                l2reg=train_l2reg,
                                lossfunc=train_lossfunc)
        saved_random_weights = model_.get_weights()

    # train 1st model with 1st optimizer (for weight initialization)        
    model_.set_weights(saved_random_weights)
    model_.fit(X_train_, Y_train_,
               validation_data=(X_valid_, Y_valid_),
               batch_size=MinBatchSize,
               epochs=num_epoch_2nd,
               verbose=0,
               callbacks=[EarlyStopping(monitor='val_r_square', patience=patience_2nd,
                                        mode='max', restore_best_weights=True),
                          cb_PrintDot()
                          ],
               shuffle=False
              )

    # train 2nd (actual) model 
    model.set_weights(model_.get_weights())
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
        selected_epoch = os.listdir(checkpoint_dir)[-1]
    elif early_stop_metric == "R2 rolling mean":
        model_selected, selected_epoch = selectmodel_basedon_movMetric(X_valid_, Y_valid_, mov_windowsSize,
                     history, checkpoint_dir, checkpoint_filesuffix, model, metric='val_r_square')
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
        model_temp.load_weights(savepath_models+'\\'+modelfile)
        pred_temp = model_temp.predict(X_valid_)
        pred_temp = pred_temp[abridged_validationSize * -1:]
        pred_ground = Y_valid_[abridged_validationSize * -1:]
        r2_temp = r2_score(pred_ground, pred_temp)
        r2_hist.append(r2_temp)

    max_tmp = max(r2_hist)
    max_idxs_tmp = [ii for ii, vv in enumerate(r2_hist) if vv == max_tmp]
    max_idx_tmp = max_idxs_tmp[-1]
    model_for_this_walk.load_weights(savepath_models+'\\'+model_list[max_idx_tmp])

    # get accuracy metric on validation data
    preds_valid = model_for_this_walk.predict(X_valid_)
    preds_valid = preds_valid * sigma + mu
    preds_valid = preds_valid.reshape([len(preds_valid), 1])
    preds_tdates_valid = tdates_valid
    preds_ground_valid = np.array(Y_valid_)
    preds_ground_valid = preds_ground_valid * sigma + mu
    preds_ground_valid = preds_ground_valid.reshape([len(preds_ground_valid), 1])
    r2_valid = r2_score(preds_ground_valid, preds_valid) # should be same as max_tmp
    acc_valid = accuracy_score(np.sign(preds_ground_valid), np.sign(preds_valid))

    # make prediction with selected model on test dataset----------------------------------------
    preds = model_for_this_walk.predict(X_test_)
    preds = preds[interval * -1:]
    preds = preds * sigma + mu
    preds = preds.reshape([len(preds), 1])
    preds_tdates = tdates_test[interval * -1:]
    preds_ground = np.array(Y_test_[interval * -1:])
    preds_ground = preds_ground * sigma + mu
    preds_ground = preds_ground.reshape([len(preds_ground), 1])
    r2_test = r2_score(preds_ground, preds)
    acc_test = accuracy_score(np.sign(preds_ground), np.sign(preds))

    # plots--------------------------------------------------------------------------------------
    plt.figure(figsize=[8.27,11.69])
    plt.subplot(311)
    plt.plot(history.history['r_square'])
    plt.plot(history.history['val_r_square'])
    plt.title(preds_tdates[0])
    plt.ylabel('R^2')
    # plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.subplot(312)
    xx_axis = list(range(1,len(preds_tdates_valid)+1))
    plt.xticks(xx_axis,preds_tdates_valid, rotation=70)
    plt.stem(xx_axis, preds_valid, '.','x')
    plt.stem(xx_axis, preds_ground_valid, '.','o')
    r2_valid_ = "{:.2f}".format(r2_valid)
    acc_valid_ = "{:.2f}".format(acc_valid)
    r2_test_ = "{:.2f}".format(r2_test)
    acc_test_ = "{:.2f}".format(acc_test)
    plt.title('Selected epoch: ' + selected_epoch + ',  val r2:' + r2_valid_ + \
              ',  val acc:' + acc_valid_ + ',  test r2:' + r2_test_ + ',  test acc:' + acc_test_)
    plt.legend(['pred_val', 'actual_val'], loc='upper left')
    plt.subplot(313)
    xxx_axis = list(range(1,len(preds_tdates)+1))
    plt.xticks(xxx_axis,preds_tdates, rotation=70)
    plt.stem(xxx_axis, preds, '.','x')
    plt.stem(xxx_axis, preds_ground, '.','o')
    plt.legend(['pred', 'actual'], loc='upper left')
    img_id = 'interval'+str(i)+'.png'
    if not os.path.isdir(savepath_imgs):
        os.mkdir(savepath_imgs)
    plt.savefig(savepath_imgs+'\\'+img_id)
    # plt.show()
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
            metric_writer.writerow(['period start','period length','validation r2',
                                    'test r2', 'validation acc', 'test acc'])
            metric_writer.writerow([preds_tdates[0], len(preds), r2_valid,
                                    r2_test, acc_valid, acc_test])
    else:
        with open(savepath_metrics, 'a', newline='') as csvfile:
            metric_writer = csv.writer(csvfile, delimiter=',')
            metric_writer.writerow([preds_tdates[0], len(preds), r2_valid,
                                    r2_test, acc_valid, acc_test])

    # write model configuration once for the loop------------------------------------------------
    if i == start:
        with open(savepath_config1, 'wb') as cf:
            pickle.dump(model_.get_config(), cf, pickle.HIGHEST_PROTOCOL)
        with open(savepath_config2, 'wb') as cf:
            pickle.dump(model.get_config(), cf, pickle.HIGHEST_PROTOCOL)

    # load saved model configuration if needed----------------------------------------------------
    # with open(savepath_config1, 'rb') as cf:
    #     saved_model_configuration_model1 = pickle.load(cf)
    # with open(savepath_config2, 'rb') as cf:
    #     saved_model_configuration_model2 = pickle.load(cf)    

    # del model, model_for_this_walk, model_selected, model_temp, history
    # del X_cut, Y_cut, tdates_cut, X_wf, Y_wf, tdates_wf, \
    #     Xdim, validation_idx, X_train_, Y_train_, X_valid_, Y_valid_, tdates_valid, \
    #     X_test_, Y_test_, tdates_test, logdir_run, cb_trainstop, \
    #     CBs, model_id, model_list, r2_hist, modelfile, pred_temp, pred_ground, \
    #     r2_temp, max_tmp, max_idxs_tmp, max_idx_tmp, preds, preds_tdates, \
    #     preds_ground, r2_test, xx_axis, img_id, dataf, dataF#, cb_tensorboard

# %% save predictions as mat file

csvfile.close()
savefile_mat = saveDIR_root + '\\' + tpName + '_' + time_stamp + '.mat'
spio.savemat(savefile_mat, {'struct': wfpreds.to_dict("list")})