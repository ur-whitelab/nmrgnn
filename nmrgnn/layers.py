from tensorflow import keras
import tensorflow as tf


class MPLayer(keras.layers.Layer):
    def __init__(self, activation=None, name='MPLayer'):
        super(MPLayer, self).__init__(name=name)
        self.activation = activation

    def build(self, input_shape):
        node_feature_shape, _, edge_feature_shape = input_shape
        self.w = self.add_weight(
            shape=(node_feature_shape[-1], node_feature_shape[-1], edge_feature_shape[-1]),
            trainable=True
        )

    def call(self, inputs):
        # node ->  N x node_feature
        # nlist -> N x NN 
        nodes, nlist, edges = inputs
        # Get node matrix sliced by nlist -> N x NN x node_features
        sliced_features = tf.gather(nodes, nlist)
        prod = tf.einsum('ijn,ijl,lmn->ijm', edges, sliced_features, self.w)
        # now we pool across neighbors
        reduced = tf.reduce_mean(prod, axis=1)
        if self.activation is not None:
            out = self.activation(reduced)
        else:
            out = reduced
        # output -> N x D number of atoms x node feature dimension
        return out
