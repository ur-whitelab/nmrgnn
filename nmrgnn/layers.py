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


class RBFExpansion(tf.keras.layers.Layer):
    R''' A  continuous-filter convolutional radial basis filter input from
    `SchNet <https://arxiv.org/pdf/1706.08566.pdf>`_.
    The input should be a rank ``K`` tensor of distances. The output will be rank ``K``
    with the new axis being of dimension ``count``. The distances are converted with
    :math:`\exp\gamma\left(d - \mu\right)^2` where :math:`\mu` is an evenly spaced
    grid from ``low`` to ``high`` containing ``count`` elements. The distance between
    elements is :math:`1 / \gamma`.
    '''

    def __init__(self, low, high, count):
        R'''
        :param low: lowest :math:`\mu`
        :type low: float
        :param high: high :math:`\mu` (inclusive)
        :type high: float
        :param count: Number of elements in :math:`\mu` and output last axis dimension
        :type count: int
        '''
        super(RBFExpansion, self).__init__(name='rbf-layer')
        self.low = low
        self.high = high
        self.centers = tf.cast(tf.linspace(low, high, count), dtype=tf.float32)
        self.gap = self.centers[1] - self.centers[0]

    def call(self, inputs):
        rbf = tf.math.exp(-(inputs[..., tf.newaxis] -
                            self.centers)**2 / self.gap)
        return rbf
