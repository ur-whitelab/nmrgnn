import tensorflow as tf


def corr_coeff(x, y, w=None):
    if w is None:
        w = tf.ones_like(x)
    m = tf.reduce_sum(w)
    xm = tf.reduce_sum(w * x) / m
    ym = tf.reduce_sum(w * y) / m
    xm2 = tf.reduce_sum(w * x**2) / m
    ym2 = tf.reduce_sum(w * y**2) / m
    cov = tf.reduce_sum(w * (x - xm) * (y - ym)) / m
    cor = cov / tf.math.sqrt((xm2 - xm**2) * (ym2 - ym**2))
    return cor


class NameMAE(tf.keras.metrics.Metric):
    '''Compute mean absolute error for specific atom name'''

    def __init__(self, label_name,  embeddings, regex=False, name='name-specific-loss', **kwargs):
        super(NameMAE, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='MAE', initializer='zeros')
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
        self.label_name = tf.constant(ln, dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # mask diff by which predictions match the label
        mask = tf.cast(tf.reduce_any(
            tf.equal(tf.cast(y_true[:, 1][..., tf.newaxis], self.label_name.dtype), self.label_name), axis=-1), tf.float32)
        diff = tf.math.abs(y_true[:, 0] - y_pred) * mask
        if sample_weight is not None:
            mask *= sample_weight
            diff *= sample_weight
        N = tf.reduce_sum(mask)
        self.mae.assign(tf.reduce_sum(diff) / N)

    def result(self):
        return self.mae

    def reset_states(self):
        self.mae.assign(0)


class NameR2(tf.keras.metrics.Metric):
    '''Compute mean absolute error for specific atom name'''

    def __init__(self, label_name,  embeddings, regex=False, name='name-specific-loss', **kwargs):
        super(NameR2, self).__init__(name=name, **kwargs)
        self.r2 = self.add_weight(name='R2', initializer='zeros')
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
        self.label_name = tf.constant(ln, dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # mask diff by which predictions match the label
        mask = tf.cast(tf.reduce_any(
            tf.equal(tf.cast(y_true[:, 1][..., tf.newaxis], self.label_name.dtype), self.label_name), axis=-1), tf.float32)
        if sample_weight is not None:
            mask *= sample_weight
        r = corr_coeff(y_true[:, 0], y_pred, mask)
        self.r2.assign(r**2)

    def result(self):
        return self.r2

    def reset_states(self):
        self.r2.assign(0)
