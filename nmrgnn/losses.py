import tensorflow as tf


class MeanSquaredLogartihmicErrorNames(tf.keras.losses.MeanSquaredLogarithmicError):
    def call(self, y_true, y_pred):
        return super().call(y_true[:, 0], y_pred)
