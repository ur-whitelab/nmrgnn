import tensorflow as tf
import numpy as np


def type_mask(label_name,  embeddings, regex=False):
    '''Get a list of ints that match a label_name from embeddings'''
    ln = []
    if regex:
        import re
        m = re.compile(label_name)
        for k, v in embeddings['name'].items():
            if m.match(k):
                ln.append(v)
        if len(ln) == 0:
            raise ValueError(
                'Regular expression did not match any embeddings')
    else:
        ln = [embeddings['name'][label_name]]
    return ln


class NameRMSD(tf.keras.metrics.Metric):
    '''Compute mean absolute error for specific atom name'''

    def __init__(self, label_idx, name='name-specific-loss', **kwargs):
        super(NameRMSD, self).__init__(name=name, **kwargs)
        self.rmsd = self.add_weight(name='RMSD', initializer='zeros', shape=())
        self.label_idx = label_idx
        self.ln = np.array(label_idx, dtype=np.int32)

    def get_config(self):
        config = super(NameRMSD, self).get_config()
        config.update({'label_idx': self.label_idx})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        # mask diff by which predictions match the label
        mask = y_true[:, -1] * tf.cast(tf.reduce_any(
            tf.equal(tf.cast(y_true[:, 1][..., tf.newaxis], self.ln.dtype), self.ln), axis=-1), tf.float32)
        diff = (y_true[:, 0] - y_pred)**2 * mask
        N = tf.reduce_sum(mask)
        self.rmsd.assign(tf.math.sqrt(tf.math.divide_no_nan(tf.reduce_sum(diff), N)))

    def result(self):
        return self.rmsd

    def reset_states(self):
        self.rmsd.assign(0)

class NameCount(tf.keras.metrics.Metric):
    '''Count occurences of name'''

    def __init__(self, label_idx, name='avg-name-count', **kwargs):
        super(NameCount, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros', shape=())
        self.label_idx = label_idx
        self.ln = np.array(label_idx, dtype=np.int32)

    def get_config(self):
        config = super(NameCount, self).get_config()
        config.update({'label_idx': self.label_idx})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        # mask diff by which predictions match the label
        mask = y_true[:, -1] * tf.cast(tf.reduce_any(
            tf.equal(tf.cast(y_true[:, 1, tf.newaxis], self.ln.dtype), self.ln), axis=-1), tf.float32)
        N = tf.reduce_sum(mask)
        self.count.assign(N)

    def result(self):
        return self.count

    def reset_states(self):
        self.count.assign(0)


class NameCorr(tf.keras.metrics.Metric):
    '''Compute mean absolute error for specific atom name'''

    def __init__(self, label_idx, name='name-specific-r', **kwargs):
        super(NameCorr, self).__init__(name=name, **kwargs)
        self.r = self.add_weight(name='Corr', initializer='zeros', shape=())
        self.label_idx = label_idx
        self.ln = np.array(label_idx, dtype=np.int32)

    def get_config(self):
        config = super(NameCorr, self).get_config()
        config.update({'label_idx': self.label_idx})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        # mask diff by which predictions match the label
        mask = y_true[:, -1] * tf.cast(tf.reduce_any(
            tf.equal(tf.cast(y_true[:, 1][..., tf.newaxis], self.ln.dtype), self.ln), axis=-1), tf.float32)
        r = NameCorr.corr_coeff(y_true[:, 0], y_pred, mask)
        self.r.assign(r)

    def result(self):
        return self.r

    def reset_states(self):
        self.r.assign(0.)

    @staticmethod
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
