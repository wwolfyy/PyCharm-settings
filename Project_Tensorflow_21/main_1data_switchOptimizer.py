# %% THIS IS WORK IN PROGRESS
# START TRAINING WITH NADAM AND SWITCH TO SGD

# %% import base packages

import tensorflow as tf
import numpy as np
import math
import operator
import datetime
import csv
import pandas as pd
from scipy import mean
import shutil
import os

import time
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
#from tensorflow.keras import models
#from tensorflow.keras import layers
#from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import BatchNormalization

print(tf.__version__)
print(tf.keras.__version__)

# %% check GPU & control gpu memory usage -- not necesary for windows 10 (?)

device_lib.list_local_devices()
if not tf.config.gpu.get_per_process_memory_growth():
    tf.config.gpu.set_per_process_memory_fraction(0.7)
    tf.config.gpu.set_per_process_memory_growth(True)

print(tf.config.gpu.get_per_process_memory_fraction())
print(tf.config.gpu.get_per_process_memory_growth())
#tf.debugging.set_log_device_placement(True)
#tf.config.set_soft_device_placement(True)

# %% import data

# Linux
# path_CSV = '/home/lstm/Desktop/GoogleDrive_local/Data Release/CSV'
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files 8 multiple 2 dim' # Linux
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files 8 multiple all dim'
# path_mat = '/home/lstm/Desktop/GoogleDrive_local/Data Release/mat files temp'

# 8700k
path_CSV = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\CSV' # 8700k
# path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro/Quant Project\Outsourcing/Data Release\mat files 8 multiple all dim'
path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files 8 multiple 2 dim'
# path_mat = r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\mat files temp'

# Windows on Unbuntu01
# path_CSV = r'C:\Users\jp\Google Drive\Data Release\CSV'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple all dim'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files 8 multiple 2 dim'
# path_mat = r'C:\Users\jp\Google Drive\Data Release\mat files temp'

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

TestSize = 0
MinBatchMultiple = 3 # mutiple of batchSize_multiple (usually 8); use default if 0
ValidBatchMultiple = 3 # must be multiple of MinBatchMultiple; use default if 0

# from import_data import Import3Ddata
# X_train, Y_train, X_valid, Y_valid, X_test, Y_test, batchSize_multiple, tradedates, mu, sigma = \
#     Import3Ddata(tpName, platform, MinBatchMultiple, ValidBatchMultiple, TestSize)
# imported_3d = Import3Ddata(tpName, platform, MinBatchMultiple, ValidBatchMultiple, TestSize)

from import_data import ImportCSVdata_processed
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

# %% Custom metrics

def r_square(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

def propAcc(y_true, y_pred):
    from tensorflow.keras import backend as K
    y_true = K.sign(y_true)
    y_pred = K.sign(y_pred)
    numel = K.sum(K.ones_like(y_true))
    numel = tf.cast(numel, tf.float32)
    hits = K.equal(y_true, y_pred)
    hits = tf.cast(hits, tf.float32)
    hits = K.sum(hits)
    return hits/numel

# %% set up mixed precision training

#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
#tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
# tf.keras.mixed_precision.experimental.set_policy('infer')
# print(tf.keras.mixed_precision.experimental.global_policy())

X_train = tf.cast(X_train, tf.float32)
Y_train = tf.cast(Y_train, tf.float32)
X_valid = tf.cast(X_valid, tf.float32)
Y_valid = tf.cast(Y_valid, tf.float32)
X_test = tf.cast(X_test, tf.float32)
Y_test = tf.cast(Y_test, tf.float32)

# %% construct layers and compile

def create_model(minibatch_shape, train_optimizer, activationF=None,
                 LSTMinputDroprate=0, LSTMrecDroprate=0, droprate=0,
                 l2reg=0.0001, lossfunc='mean_squared_error'):

    model = Sequential([
        LSTM(60,
             # kernel_initializer = tf.keras.initializers.glorot_normal(seed=None),
             implementation=1,
             return_sequences=True,
             return_state=False,
             stateful=True, batch_input_shape=minibatch_shape,
             unroll=False,
             dropout=LSTMinputDroprate, # dropout on LSTM cell input
             recurrent_dropout=LSTMrecDroprate, # dropout on LSTM cell recurrent input
             # activity_regularizer=regularizers.l2(l2reg),
             kernel_regularizer=regularizers.l2(l2reg),
             recurrent_regularizer=regularizers.l2(l2reg)
             ),
        BatchNormalization(),
        Activation(activationF),
        # Dropout(droprate), # dropout on layer output

        LSTM(30,
             # kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
             implementation=1,
             return_sequences=False,
             return_state=False,
             stateful=True, batch_input_shape=minibatch_shape,
             unroll=False,
             dropout=LSTMinputDroprate, # dropout on LSTM cell input
             recurrent_dropout=LSTMrecDroprate, # dropout on LSTM cell recurrent input
             # activity_regularizer=regularizers.l2(l2reg),
             kernel_regularizer=regularizers.l2(l2reg),
             recurrent_regularizer=regularizers.l2(l2reg)
             ),
        BatchNormalization(),
        Activation(activationF),
        # Dropout(droprate), # dropout on layer output

        # Dense(units=480,
              #kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
              # kernel_regularizer=regularizers.l2(l2reg)),
        # BatchNormalization(),
        # Activation(activationF),
        # Dropout(droprate),

        Dense(units=128,
              # kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
              # activity_regularizer=regularizers.l2(l2reg),
              kernel_regularizer=regularizers.l2(l2reg)),
        # BatchNormalization(),
        # Activation(activationF),
        Dropout(droprate),

        Dense(units=output_unit,
              # kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
              # activity_regularizer=regularizers.l2(l2reg),
              activation = tf.keras.activations.linear,
              kernel_regularizer=regularizers.l2(l2reg)),
    ])

    model.compile(optimizer=train_optimizer,
                  loss=lossfunc,
                  # sample_weight_mode='temporal',
                  metrics=[r_square, propAcc, lossfunc]
                  )

    return model

# %% set up callbacks (except tensorboard)

# callback: save on checkpoint
checkpoint_dir = r'C:\Users\jp\TF_checkpoints'
checkpoint_modifier = '\cp-{epoch:04d}'
checkpoint_filesuffix = '.hdf5'
checkpoint_path = checkpoint_dir + checkpoint_modifier + checkpoint_filesuffix
cb_checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            period = 1,
            verbose=0
            )

# callback: print dot
class cb_PrintDot(callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print(str(epoch))
        print('.', end='', flush=True)

# callback: mov10 R^2
def create_mov10R2_callback(stop_patience=1000, min_epochs=100, logdir=""):
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    val_r2_log = []
    val_r2_mov10_log = []

    class cb_earlystop_validR2(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}, stop_patience=stop_patience):
            val_r2_log.append(logs.get('val_r_square'))
            print(logs.get('val_r_square'))
            if epoch > 9:
                val_r2_mov10_log.append(mean(val_r2_log[-10:]))
                tf.summary.scalar('R2_mov10', data=val_r2_mov10_log[-1], step=epoch)
                if len(val_r2_mov10_log) > min_epochs:
                    maxIDX, maxVALUE = max(enumerate(val_r2_mov10_log[min_epochs:]), key=operator.itemgetter(1))
                    if (len(val_r2_mov10_log) - (maxIDX+min_epochs)) > stop_patience:
                        print('Stopping training. Max mov10 R^2 acheived '+
                              str(stop_patience)+' epochs (plus min epochs) ago')
                        print('epoch: '+str(epoch))
                        self.model.stop_training = True

    return cb_earlystop_validR2()

# %% callback: switch optimizer

# class OptimizerChanger(tf.keras.callbacks.EarlyStopping):
#     def __init__(self, on_train_end, **kwargs):
#         # self.do_on_train_end = on_train_end
#         super(OptimizerChanger, self).__init__(**kwargs)
#     def on_train_end(self, logs=None):
#         super(OptimizerChanger, self).on_train_end(self, logs)
#         self.do_on_train_end()
#
# def do_after_training():
#     print('Switching optimizer to SGD')
#     model.compile(
#         optimizer=optimizers.SGD(clipnorm=1., decay=0., momentum=0.),
#         loss = 'logcosh',
#         metrics=[r_square, propAcc, lossfunc]
#     )
#     model.fit(X_train, Y_train,
#             validation_data=(X_valid, Y_valid),
#             batch_size=MinBatchSize,
#             epochs=num_epoch,
#             verbose=0,
#             callbacks=CBs,
#             shuffle=False
#             )
# cb_optimizer_changer = OptimizerChanger(
#     on_train_end=do_after_training,
#     monitor='val_r_square',
#     min_delta=1e-10,
#     patience=20,
#     restore_best_weights=True



# %% define params

# train_optimizer = [optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False, clipnorm=1.),'']
train_optimizer = [
#     optimizers.Adadelta(clipnorm=1.),
#     [optimizers.SGD(clipnorm=1.),'_vanila'],
#     [optimizers.SGD(clipnorm=1., momentum=0., nesterov=True), '_Nstrv'],
#     [optimizers.SGD(clipnorm=1., momentum=0.25, nesterov=True), '_m025_Nstrv'],
#     [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=True), '_m050_Nstrv'],
#     [optimizers.SGD(clipnorm=1., momentum=0.75, nesterov=True), '_m075_Nstrv'],
    [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=True), '_m090_Nstrv'],
    # [optimizers.SGD(clipnorm=1., momentum=0., nesterov=False), '_vanila'],
    # [optimizers.SGD(clipnorm=1., momentum=0.25, nesterov=False), '_m025'],
    # [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=False), '_m050'],
    # [optimizers.SGD(clipnorm=1., momentum=0.75, nesterov=False), '_m075'],
    # [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=False), '_m090'],
#     [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=True), '_m05_Nstrv'],
#     [optimizers.SGD(clipnorm=1., momentum=0.5, nesterov=False), '_m05'],
#     [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=True), '_m09_Nstrv'],
#     [optimizers.SGD(clipnorm=1., momentum=0.9, nesterov=False), '_m09']
#     optimizers.RMSprop(clipnorm=1.),
#     optimizers.Adagrad(clipnorm=1.),
#     optimizers.Adamax(clipnorm=1.),
#     [optimizers.Nadam(clipnorm=1.),'_vanila']
#     optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False, clipnorm=1.)
]

# train_activation = [tf.nn.elu]
train_activation = [
    # tf.nn.leaky_relu,
    # tf.nn.relu,
    # tf.nn.relu6,
    # tf.nn.selu,
    tf.nn.elu
]

train_lossfunc = [losses.LogCosh()]
# train_lossfunc = [
#     losses.MeanSquaredError(),
#     losses.MeanAbsoluteError(),
#     losses.Huber(delta=1.0),
#     losses.LogCosh()
# ]

train_LSTMinputDroprate = 0.5
train_LSTMrecDroprate = 0.0
train_droprate = 0.5
train_l2reg = 0.0001

MinBatchSize= MinBatchMultiple * batchSize_multiple
num_epoch = 2000
minimum_epochs = 50
output_unit = 1

Xdim = X_train.shape
minibatch_shape= [MinBatchSize, Xdim[1], Xdim[2]]
logdir_root=r'C:\Users\jp\TF_logs\log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# %% initialize wieghts w/ Nadam optimizer

# def initialize_custom_weights(checkpoint_path_iw, model_, X_train, Y_train, X_valid, Y_valid, MinBatchSize):
#     # with tf.device("GPU:0"):
#     model_.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=MinBatchSize, epochs=300, verbose=0,
#         callbacks=[callbacks.EarlyStopping(monitor='r_square', patience=20, restore_best_weights=True)],
#         shuffle=False
#         )
#     return model_

# %% fit model

print('Run: tensorboard --logdir now:'+logdir_root)
input("Press Enter to continue...")

for i in range(len(train_optimizer)):
    print(i)
    train_optimizer_ = train_optimizer[i][0]
    optimizer_name = train_optimizer_.__dict__.get("_name") + train_optimizer[i][1]
    for train_activation_ in train_activation:
        activation_name = train_activation_.__name__
        for train_lossfunc_ in train_lossfunc:
            lossfunc_name = train_lossfunc_.__dict__.get("name")

            logdir_run=r'\\'+optimizer_name+'_'+activation_name+'_'+lossfunc_name
            cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir_root+logdir_run, histogram_freq=0)
            cb_earlystop = create_mov10R2_callback(stop_patience=300, min_epochs=100, logdir=logdir_root+logdir_run)
            CBs = [
                cb_tensorboard,
                cb_checkpoint,
                cb_PrintDot(),
                cb_earlystop
            ]

            # initialize weights w/ Nadam optimizer
            model_ = create_model(minibatch_shape, optimizers.Nadam(), train_activation_,
                                 train_LSTMinputDroprate, train_LSTMrecDroprate, train_droprate,
                                 train_l2reg, train_lossfunc_)
            model_.fit(X_train, Y_train,
                      validation_data=(X_valid, Y_valid),
                      batch_size=MinBatchSize,
                      epochs=200,
                      verbose=2,
                      callbacks=[EarlyStopping(monitor='val_r_square', patience=25,
                                               mode='max', restore_best_weights=True)],
                      shuffle=False
                      )

            # create model and load intial weights
            model = model_.from_config(model_.get_config())
            model.set_weights(model_.get_weights())
            model.compile(optimizer=train_optimizer_,
                          loss=train_lossfunc_,
                          # sample_weight_mode='temporal',
                          metrics=[r_square, propAcc, train_lossfunc_]
                          )
            # model = create_model(minibatch_shape, train_optimizer_, train_activation_,
            #                      train_LSTMinputDroprate, train_LSTMrecDroprate, train_droprate,
            #                      train_l2reg, train_lossfunc_)
            # model.load_weights(checkpoint_path_iw)

            # with tf.device("GPU:0"):
            history = model.fit(X_train, Y_train,
                                validation_data=(X_valid, Y_valid),
                                batch_size=MinBatchSize,
                                epochs=num_epoch,
                                verbose=0,
                                callbacks = CBs,
                                shuffle=False
                                )

            del model
            del model_

# %% select model from checkpoints

R2_windowsSize = 3
R2_windowOffset = math.floor(R2_windowsSize/2)
def runningMean_validOnly(x, N):
    avg_mask = np.ones(N) / N
    return np.convolve(x, avg_mask, 'valid')

R2_mov3 = runningMean_validOnly(history.history['val_r_square'],R2_windowsSize) # get 3-day moving avg of r2
maxR2idx, maxR2value = max(enumerate(R2_mov3[minimum_epochs:]), key=operator.itemgetter(1))
maxR2idx = maxR2idx + R2_windowOffset + minimum_epochs + 1 # checkpoint starts at 1
maxR2range = list(range(maxR2idx-R2_windowOffset, maxR2idx+R2_windowOffset+1))

model_tmp = create_model(MinBatchSize, Xdim[1], Xdim[2], train_optimizer, train_activation,
                     train_LSTMinputDroprate, train_LSTMrecDroprate, train_droprate,
                     train_l2reg, train_lossfunc)
val_r2list = list()
val_r2list = []
for i in range(len(maxR2range)):
    maxR2idx_str = '\cp-'+"{0:0=4d}".format(maxR2range[i])
    selectModel_path = checkpoint_dir + maxR2idx_str + checkpoint_filesuffix
    model_tmp.load_weights(selectModel_path, by_name=False)
    results_tmp = model_tmp.evaluate(X_valid, Y_valid)
    val_r2list.append(results_tmp[1])

maxR2idx_inRange, maxR2value = max(enumerate(val_r2list), key=operator.itemgetter(1))
maxR2idx_final_str = '\cp-'+"{0:0=4d}".format(maxR2range[maxR2idx_inRange]) # index in checkpoint name, not in history.history

print( 'Selection: '+maxR2idx_final_str+' out of'+str(maxR2range) )
answer = input('Accept selected model? [Y/N]\n')
if answer == 'Y':
    print('yes')
else:
    print('no')
selectModel_path = checkpoint_dir + maxR2idx_final_str + checkpoint_filesuffix

model_selected = create_model(MinBatchSize, Xdim[1], Xdim[2], train_optimizer, train_activation,
                     train_LSTMinputDroprate, train_LSTMrecDroprate, train_droprate,
                     train_l2reg, train_lossfunc)
model_selected.load_weights(selectModel_path, by_name=False)
savemodel_path = r'C:\Users\jp\TF_saved_models'
i = 1
model_name = tpName+'_'+str(i)+'.h5'
model_selected.save(savemodel_path + model_name)

# %% make prediction with selected model on test dataset

X_test_temp = np.tile(X_test,[batchSize_multiple,1])
X_test_temp = X_test_temp[MinBatchSize * -1:]

preds = model_selected.predict(X_test_temp)
#preds = np.concatenate( preds, axis=0 )
preds = np.vstack(preds)
preds = preds[interval * -1:]
preds = preds * sigma + mu
tdates = tdates_cut[interval * -1:]
print(tdates)

dataf = {'tradedate':tdates.tolist(), 'prediction':preds.tolist()}
dataF = pd.DataFrame.from_dict(dataf)
wfpreds = pd.concat([wfpreds, dataF])

# %% leftover codes: post predcition for current walk: save in CSV; clear checkpoint folder; reset logs

#     saveDIR = r'C:\Users\jp\Google Drive\JP data on ubuntu01\WFpredictions_WIP\\' # Windows
#     #saveDIR = r'F:WFpredictions_WIP\\' # Windows on Ubuntu
#     save_id = tp_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     savefile_csv = saveDIR +save_id + '.csv'
#     wfpreds.to_csv(savefile_csv, index=False)

#     shutil.rmtree(checkpoint_dir)
#     time.sleep(5)
#     os.makedirs(checkpoint_dir)

#     val_r2_log = []
#     val_r2_mov5_log = []

# # %% save predictions as mat file

# savefile_mat = saveDIR + save_id + '.mat'
# spio.savemat(savefile_mat, {'struct':wfpreds.to_dict("list")})
