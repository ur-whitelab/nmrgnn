import tensorflow as tf
import numpy as np

def corr_coeff(x, y, w = None):
    if w is None:
        w = tf.ones_like(x)
    m = tf.reduce_sum(w)
    xm = tf.reduce_sum(w * x) / m
    ym = tf.reduce_sum(w * y) / m
    xm2 = tf.reduce_sum(w * x**2) / m
    ym2 = tf.reduce_sum(w * y**2) / m
    cov = tf.reduce_sum( w * (x - xm) * (y - ym) )
    # clip because somehow we get negative covariance sometimes (???)
    cor = tf.math.divide_no_nan(cov, m * tf.math.sqrt(tf.clip_by_value((xm2 - xm**2) * (ym2 - ym**2), 0, 1e32)))
    return cor

class NameLoss(tf.keras.losses.Loss):
    '''Compute L2 loss * s + corr_loss * (1 - s) for specific atom name'''
    def __init__(self, label_idx, s=1., name='name-loss', reduction='none'):
        super(NameLoss, self).__init__(name=name, reduction=reduction)
        self.label_idx = label_idx
        self.ln = np.array(label_idx, dtype=np.int32)
        self.s = s

    def get_config(self):
        config = {'label_idx': self.label_idx, 's': self.s}
        base_config = super(NameLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, y_true, y_pred, sample_weight=None):
        # mask diff by which predictions match the label

        w = y_true[:, -1] * tf.cast(tf.reduce_any(
            tf.equal(tf.cast(y_true[:, 1][..., tf.newaxis], self.ln.dtype), self.ln), axis=-1), tf.float32)
        x = y_pred
        y = y_true[:,0]
        l2 = tf.math.divide_no_nan(tf.reduce_sum( w * ( y - x)**2 ), tf.reduce_sum(w))
        r = corr_coeff(x, y, w)
        return l2 * self.s + (1 - self.s) * (1 - r)
        

