import numpy as np
from keras.callbacks import Callback
from tf.keras.losses import MeanAbsoluteError

class nonzero_MAE(Callback):
    """
    metric callback inspired by 
        https://stackoverflow.com/questions/51728648/how-do-masked-values-affect-the-metrics-in-keras
    """
   def on_train_begin(self, logs={}):
       self.loss_f = MeanAbsoluteError()
       self.val_res = []

   def on_epoch_end(self, epoch, logs={}):
       val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
       val_targ = self.model.validation_data[1] 
       indx = np.where(~val_targ.any(axis=2))[0] #find where all targets are zero. That are the masked once as we masked the target with 0 and the data with 666
       y_true_nomask = numpy.delete(val_targe, indx, axis=0)
       y_pred_nomask = numpy.delete(val_predict, indx, axis=0)

       result = self.loss_f(y_true_nomask, y_pred_nomask)
       self.val_res.append(result)

       print “ — non-zero MAE: %f ” %( result )
       return