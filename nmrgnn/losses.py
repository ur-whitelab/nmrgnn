import tensorflow as tf


class MeanSquaredLogartihmicErrorNames(tf.keras.losses.MeanSquaredLogarithmicError):
    def call(self, y_true, y_pred):
        return super().call(y_true[:, 0], y_pred)


def corr_coeff(x, y, w = None):
    if w is None:
        w = tf.ones_like(x)
    m = tf.reduce_sum(w)
    xm = tf.reduce_sum(w * x) / m
    ym = tf.reduce_sum(w * y) / m
    xm2 = tf.reduce_sum(w * x**2) / m
    ym2 = tf.reduce_sum(w * y**2) / m
    cov = tf.reduce_sum( w * (x - xm) * (y - ym) )
    cor = tf.math.divide_no_nan(cov, m * tf.math.sqrt((xm2 - xm**2) * (ym2 - ym**2)))
    return cor

def corr_loss(labels, predictions, sample_weight = None, s=1e-2):
    '''
    Mostly correlation, with small squared diff
    '''
    x = predictions
    y = labels[:,0]
    w = labels[:,-1]
    l2 = tf.math.divide_no_nan(tf.reduce_sum( w * tf.math.abs( y - x) ), tf.reduce_sum(w))
    return s * l2 + (1 - corr_coeff(x, y, w)**2)


