from tensorflow import keras
import tensorflow as tf


class MPLayer(keras.layers.Layer):
    def __init__(self, activation=None, name='MPLayer'):
        super(MPLayer, self).__init__(name=name)
        self.activation = activation

    def build(self, input_shape):
        node_shape, nlist_shape, edge_shape = input_shape
        node_feature_shape = node_shape[-1]
        edge_feature_shape = edge_shape[-1]
        self.w = self.add_weight(
            shape=(node_feature_shape, node_feature_shape, edge_feature_shape),
            trainable=True,
        )

    def call(self, inputs):
        nodes, nlist, edges = inputs
        # TODO add batch index
        # https://github.com/whitead/graphnmr/blob/master/graphnmr/gcnmodel.py#L824
        sliced_features = tf.gather_nd(nodes, nlist)
        prod = tf.einsum('bijn,bijl,lmn->bijm', edges, sliced_features, self.w)
        # now we pool across neighbors
        reduced = tf.reduce_mean(prod, axis=2)
        if self.activation is not None:
            out = self.activation(reduced)
        else:
            out = reduced
        return out
