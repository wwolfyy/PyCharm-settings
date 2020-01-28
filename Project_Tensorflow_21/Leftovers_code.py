# additions

# 128 features --> 122
# tensorboard callback (w/ remove prv logs code)
# 1 more dense layer
# gradient clipping
# propAcc metric
# early stop on MSE loss

class propSignAccuracy(tf.keras.metrics.Metric):
    def update_state(self, y_true, y_pred, sample_weight=None):
        from tensorflow.keras import backend as K
        y_true = K.sign(y_true)
        y_pred = K.sign(y_pred)
        numel = len(y_true)
        hits = K.equal(y_true, y_pred)
        hits = tf.cast(hits, tf.int8)
        values = hits / numel
        values = tf.cast(values, self.dtype)
        self.propSignAccuracy.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


# %% callbacks

class MyLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('log.txt', 'a+') as f:
            f.write('%02d %.3f\n' % (epoch, logs['loss']))
            
csv_logger = tf.keras.callbacks.CSVLogger('training.log')            

his_tory = tf.keras.callbacks.History()

from time import time
import datetime
logdir=r'C:\Users\jp\TF_logs\log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

logrecord = []
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      logrecord.append(logs.get('val_r_square'))
      print(logrecord)
    
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
      
callback_earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_r_square',
        min_delta=0,
        patience=500,
        verbose=0,
        mode='max',
        baseline=None,
        restore_best_weights=False
        )      

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

# early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
      
#%% ways to clear gpu memory
del model

import gc
gc.collect()

from tensorflow.keras import backend as K
K.clear_session()

from numba import cuda
cuda.select_device(0)
cuda.close()
cuda.select_device(0)

# %% Misc

    # get model configuration
    model.get_config()
    model.optimizer.get_config()
    model.loss

    # moving average
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    import numpy as np
    def runningMean_validOnly(x, N):
        avg_mask = np.ones(N) / N
        return np.convolve(x, avg_mask, 'valid')

    # write to CSV
    prediction_log_path = r'C:\Users\jp\TF_PredictionLog\prediction_log.csv'
    if i == start:
        with open('person.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)
    csvFile.close()

    # check GPU status
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    get_available_gpus()
    tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)

    # r-saured (alternative)
    def r_square(y_true, y_pred):
        import numpy as np
        import statsmodels.api as sm
        model = sm.OLS(Y,X,hasconst=True)
        results = model.fit()
        return results.rsquared

    # plot validation r-squared
    plt.plot(history.history['r_square'])
    plt.plot(history.history['val_r_square'])
    plt.title('Model R^2 on Training - HIGH')
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # make prediction w/ validation set
    model = create_model(Xdim[1], Xdim[2], train_optimizer, train_activation, train_droprate)
    selectModel_path =
    model_tmp.load_weights(selectModel_path, by_name=False)
    results_tmp = model_tmp.evaluate(X_valid, Y_valid)
    preds = model_selected.predict(X_valid)

    # save as mat file
    import scipy.io as spio
    spio.savemat(r'C:\Users\jp\Downloads\X_3d.mat', {'X':X, 'Y':Y})

    # save numpy array
    import numpy as np
    np.save(r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\np arrays\X_train',X_train)
    np.save(r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\np arrays\Y_train',Y_train)
    np.save(r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\np arrays\X_valid',X_valid)
    np.save(r'C:\Users\jp\Google Drive\Wadilt Macro\Quant Project\Outsourcing\Data Release\np arrays\Y_valid',Y_valid)

    # use TF 1.x module
    tf.compat.v1.Session(config=...)
    config = tf.ConfigProto(device_count = {'GPU': 1})
    sess = tf.Session(config=config)

    # limit gpu for current workspace -- run before importing TF
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # 1070 is on first PCI slot

