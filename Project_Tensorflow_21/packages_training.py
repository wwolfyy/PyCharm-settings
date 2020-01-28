# metric: r-squared (no intercept)
def r_square(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# metric: prop accuracy
def propAcc(y_true, y_pred):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    y_true = K.sign(y_true)
    y_pred = K.sign(y_pred)
    numel = K.sum(K.ones_like(y_true))
    numel = tf.cast(numel, tf.float32)
    hits = K.equal(y_true, y_pred)
    hits = tf.cast(hits, tf.float32)
    hits = K.sum(hits)
    return hits/numel


# create model (L:LSTM, B:batch norm, A:activation, D:dropout, F:dense, R:regression)
def create_model(minibatch_shape,
                 LSTMsize,
                 FCsize,
                 Convsize, # how many convolutional layers to put in
                 train_optimizer,
                 activationF=None,
                 LSTMinputDroprate=0,
                 LSTMrecDroprate=0,
                 droprate=0,
                 l2reg=0.0001,
                 lossfunc='mean_squared_error',
                 bi_dir = False,
                 conv_1D = False,
                 conv_2D = False,
                 conv_LSTM_2D = False,
                 conv_params = {},
                 conv_input_shape = [None, None],
                 compile_model = True):

    from tensorflow.keras import optimizers, regularizers, losses, activations
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, ConvLSTM2D, \
        BatchNormalization, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Bidirectional

    # ----------------------------------------------------------------------------------
    model = Sequential()

    # ------------------------convolution block -------------------------------------
    if conv_1D or conv_2D:

        for i in range(Convsize):

            if conv_1D:
                if i == 0:
                    model.add(Conv1D(filters=conv_params['filters'],
                                     kernel_size=conv_params['kernel'],
                                     activation=conv_params['activation'],
                                     padding = conv_params['padding'],
                                     batch_input_shape=minibatch_shape,
                                     input_shape=conv_input_shape))
                else:
                    model.add(Conv1D(filters=conv_params['filters'],
                                     kernel_size=conv_params['kernel'],
                                     padding = conv_params['padding'],
                                     activation=conv_params['activation']))
                #model.add(MaxPooling1D())

            if conv_2D:
                if i == 0:
                    model.add(Conv2D(filters=conv_params['filters'],
                                     kernel_size=conv_params['kernel'],
                                     activation=conv_params['activation'],
                                     padding = conv_params['padding'],
                                     input_shape=conv_input_shape))
                else:
                    model.add(Conv2D(filters=conv_params['filters'],
                                     kernel_size=conv_params['kernel'],
                                     padding = conv_params['padding'],
                                     activation=conv_params['activation']))

                model.add(MaxPooling2D(padding='same'))

                if i == Convsize-1:
                    model.add(Flatten())

    # ------------------------LSTM block --------------------------------------------
    if not conv_2D: # if using conv2D, LSTM is not used

        for i in range(len(LSTMsize)):

            if i < len(LSTMsize) - 1:
                output_sequence = True
            else:
                output_sequence = False

            if conv_LSTM_2D:
                model.add(ConvLSTM2D(filters=conv_params['filters'],
                                 kernel_size=conv_params['kernel'],
                                 activation=conv_params['activation'],
                                 #input_shape=conv_input_shape,
                                 kernel_regularizer = regularizers.l2(l2reg),
                                 recurrent_regularizer = regularizers.l2(l2reg),
                                 dropout = LSTMinputDroprate,  # dropout on LSTM cell input
                                 recurrent_dropout = LSTMrecDroprate,  # dropout on LSTM cell recurrent input
                                 return_sequences = output_sequence,
                                 stateful = True, batch_input_shape=conv_input_shape))
            else:
                if bi_dir:
                    model.add(Bidirectional(
                             LSTM(units=LSTMsize[i],
                                 # kernel_initializer = tf.keras.initializers.glorot_normal(seed=None),
                                 implementation=1,
                                 return_sequences=output_sequence,
                                 return_state=False,
                                 stateful=True,
                                 unroll=False,
                                 dropout=LSTMinputDroprate, # dropout on LSTM cell input
                                 recurrent_dropout=LSTMrecDroprate, # dropout on LSTM cell recurrent input
                                 # activity_regularizer=regularizers.l2(l2reg),
                                 kernel_regularizer=regularizers.l2(l2reg),
                                 recurrent_regularizer=regularizers.l2(l2reg)
                                 ),
                             batch_input_shape=minibatch_shape))
                else:
                    model.add(LSTM(units=LSTMsize[i],
                             # kernel_initializer = tf.keras.initializers.glorot_normal(seed=None),
                             implementation=1,
                             return_sequences=output_sequence,
                             return_state=False,
                             stateful=True, batch_input_shape=minibatch_shape,
                             unroll=False,
                             dropout=LSTMinputDroprate, # dropout on LSTM cell input
                             recurrent_dropout=LSTMrecDroprate, # dropout on LSTM cell recurrent input
                             # activity_regularizer=regularizers.l2(l2reg),
                             kernel_regularizer=regularizers.l2(l2reg),
                             recurrent_regularizer=regularizers.l2(l2reg)
                             ))
            model.add(BatchNormalization())
            model.add(Activation(activationF))
            # model.add(Dropout(droprate)) # dropout on layer output

    # -------------------------FC block  -----------------------------------------------
    for i in range(len(FCsize)):

        model.add(Dense(units=FCsize[i],
                  kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
                  kernel_regularizer=regularizers.l2(l2reg)))
        # model.add(BatchNormalization())
        # model.add(Activation(activationF))
        model.add(Dropout(droprate))

    # -------------------------Regression block ------------------------------------------
    model.add(Dense(units=1,
              # kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
              # activity_regularizer=regularizers.l2(l2reg),
              activation = activations.linear,
              kernel_regularizer=regularizers.l2(l2reg)))
    # ------------------------------------------------------------------------------------

    if compile_model:
        model.compile(optimizer=train_optimizer,
                      loss=lossfunc,
                      # sample_weight_mode='temporal',
                      metrics=[r_square, propAcc, lossfunc]
                      )

    return model


# callback: print dot
import tensorflow as tf
class cb_PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print(str(epoch))
        print('.', end='', flush=True)


# callback: early stop on R2 moving average
def create_movMetricStop_callback(stop_patience=500, min_epochs=100, logdir="", windowsize=7):
    import tensorflow as tf
    import operator
    from scipy import mean
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    val_r2_log = []
    val_r2_mov_log = []

    class cb_earlystop_validR2(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}, stop_patience=stop_patience):
            val_r2_log.append(logs.get('val_r_square'))
            # print(logs.get('val_r_square'))
            if epoch > windowsize:
                val_r2_mov_log.append(mean(val_r2_log[windowsize*-1:]))
                tf.summary.scalar('R2_mov', data=val_r2_mov_log[-1], step=epoch)
                if len(val_r2_mov_log) > min_epochs:
                    maxIDX, maxVALUE = max(enumerate(val_r2_mov_log[min_epochs:]), key=operator.itemgetter(1))
                    if (len(val_r2_mov_log) - (maxIDX+min_epochs)) > stop_patience:
                        print('Stopping training. Max mov R^2 acheived '+
                              str(stop_patience)+' epochs (plus min epochs) ago')
                        print('epoch: '+str(epoch))
                        self.model.stop_training = True

    return cb_earlystop_validR2()


# function: select model based on R2 moving average
def selectmodel_basedon_movMetric(X_valid, Y_valid, mov_windowsSize, history, checkpoint_dir,
                                  checkpoint_filesuffix, model, metric='val_r_square'):
    import math
    import numpy as np
    import operator

    windowOffset = math.floor(mov_windowsSize / 2)

    def runningMean_validOnly(x, N):
        avg_mask = np.ones(N) / N
        return np.convolve(x, avg_mask, 'valid')

    metric_mov = runningMean_validOnly(history.history[metric], mov_windowsSize)  # get X-day moving avg of r2
    max_idx, max_value = max(enumerate(metric_mov), key=operator.itemgetter(1))
    max_idx = max_idx + windowOffset + 1  # checkpoint starts at 1
    max_range = list(range(max_idx - windowOffset, max_idx + windowOffset + 1))

    # model_tmp = model.from_config(model.get_config())

    val_r2list = []
    for i in range(len(max_range)):
        max_idx_str = '\cp-' + "{0:0=4d}".format(max_range[i])
        Model_path = checkpoint_dir + max_idx_str + checkpoint_filesuffix
        # model_tmp.load_weights(Model_path, by_name=False)
        # results_tmp = model_tmp.evaluate(X_valid, Y_valid)
        model.load_weights(Model_path, by_name=False)
        results_tmp = model.evaluate(X_valid, Y_valid)
        val_r2list.append(results_tmp[1])

    max_value_inRange = max(val_r2list)
    idxs_tmp = [ii for ii, vv in enumerate(val_r2list) if vv == max_value_inRange]
    max_idx_inRange = idxs_tmp[-1]  # last occurrence of max value
    max_idx_final_str = '\cp-' + "{0:0=4d}".format(max_range[max_idx_inRange]) # index in checkpoint name, not in history.history
    print(max_idx_final_str)
    # answer = input('Accept selected model? [Y/N]\n')
    # if answer == 'Y':
    #     print('yes')
    # else:
    #     print('no')
    selectModel_path = checkpoint_dir + max_idx_final_str + checkpoint_filesuffix
    model.load_weights(selectModel_path, by_name=False)

    return model, max_idx_final_str