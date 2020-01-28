import matlab.engine
mpy = matlab.engine.start_matlab()
mpy.tempfunction(nargout=0)


# -----------------------------for debugging create_model function--------------------------

import tensorflow as tf
from tensorflow.keras import optimizers, regularizers, losses, activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, ConvLSTM2D, \
    BatchNormalization, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Bidirectional

minibatch_shape=[32,96,16]
LSTMsize=[240,120]
FCsize=[128,64]
#Convsize=2
train_optimizer=optimizers.Nadam(lr=0.002, clipnorm=1.)
activationF=None
LSTMinputDroprate=0
LSTMrecDroprate=0
droprate=0
l2reg=0.0001
lossfunc='mean_squared_error'

bi_dir = False
# bi_dir = True

conv_1D = False
conv_1D = True
conv_params = {'filters':108, 'kernel':1, 'activation':'relu', 'padding':'same'}
conv_input_shape = [96, 16]
Convsize=2

conv_2D = False
# conv_2D = True
# conv_params = {'filters':108, 'kernel':3, 'activation':'relu', 'padding':'same'}
# conv_input_shape = [96, 16, 1]
# Convsize=4

# conv_LSTM_2D = False
conv_LSTM_2D = True
conv_params = {'filters':108, 'kernel':3, 'activation':'relu', 'padding':'same'}
conv_input_shape = [32, 7, 96, 16, 1]

# conv_params = {},

compile_model = True

model__ = create_model(minibatch_shape,
                 LSTMsize,
                 FCsize,
                 Convsize,
                 train_optimizer,
                 activationF=None,
                 LSTMinputDroprate=0,
                 LSTMrecDroprate=0,
                 droprate=0,
                 l2reg=0.0001,
                 lossfunc='mean_squared_error',
                 bi_dir = bi_dir,
                 conv_1D = conv_1D,
                 conv_2D = conv_2D,
                 conv_LSTM_2D = conv_LSTM_2D,
                 conv_params = conv_params,
                 conv_input_shape = conv_input_shape,
                 compile_model = True)

#model__.build([128,96,16])
model__.summary()

# ---------------------------------------------------------------------------------------------










