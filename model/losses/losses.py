import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class CCE_Generalized_Dice_Loss(tf.keras.losses.Loss):
    '''
    '''
    def call(self, y_true, y_pred):
        '''
        '''
        n_classes = y_pred.shape[-1]
        w = np.zeros(shape=(n_classes,))
        w = K.sum(y_true, -1)
        w = 1/(w**2+1e-6)
        numerator = y_true*y_pred
        numerator = w*K.sum(numerator,-1)
        denominator = y_true+y_pred
        denominator = w*K.sum(denominator,-1)
        gen_dice_coef = 2*numerator/denominator
        gdl = 1 - gen_dice_coef
        cce = tf.keras.losses.categorical_crossentropy(y_true,y_pred)
        return K.mean(cce) + K.mean(gdl)