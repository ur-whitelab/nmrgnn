import tensorflow as tf


def call_mlse(self, y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError
    return msle(y_true[:, 0], y_pred)
