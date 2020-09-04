from tensorflow import keras
import tensorflow as tf


class MPLayer(keras.layers.Layer):
    def __init__(self, activation=None, name='MPLayer'):
        super(MPLayer, self).__init__(name=name)
        self.activation = activation

    def build(self, input_shape):
        node_feature_shape, _, edge_feature_shape, _ = input_shape
        self.w = self.add_weight(
            shape=(
                node_feature_shape[-1], node_feature_shape[-1], edge_feature_shape[-1]),
            trainable=True
        )

    def call(self, inputs):
        # node ->  N x node_feature
        # nlist -> N x NN
        # edges -> N x NN x edge_features
        # inv_degree -> N
        nodes, nlist, edges, inv_degree = inputs
        # Get node matrix sliced by nlist -> N x NN x node_features
        sliced_features = tf.gather(nodes, nlist)
        # i -> atom
        # j -> neighbor
        # n - > edge feature input
        # l -> atom feature input
        # m -> atom atom feature output
        reduced = tf.einsum('ijn,ijl,lmn,i->im', edges,
                            sliced_features, self.w, inv_degree)
        if self.activation is not None:
            out = self.activation(reduced)
        else:
            out = reduced
        # output -> N x D number of atoms x node feature dimension
        return out
