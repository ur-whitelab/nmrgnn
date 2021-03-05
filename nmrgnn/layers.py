from tensorflow import keras
import tensorflow as tf


class MPLayer(keras.layers.Layer):
    def __init__(self, activation=None, kernel_regularizer=None, name='MPLayer', **kwargs):
        super(MPLayer, self).__init__(name=name, **kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.mpl_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        node_feature_shape, _, edge_feature_shape, _ = input_shape
        self.w = self.add_weight(
            shape=(
                node_feature_shape[-1], node_feature_shape[-1], edge_feature_shape[-1]),
            trainable=True,
            name='w'
        )

    def get_config(self):
        config = super(MPLayer, self).get_config()
        config.update(
            {'activation': self.activation})
        return config

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
        # TODO break it up to reduce memory
        out = self.activation(reduced)
        # output -> N x D number of atoms x node feature dimension
        if self.mpl_regularizer is not None:
            self.add_loss(self.mpl_regularizer(self.w))
        return out

class AMPLayer(keras.layers.Layer):
    def __init__(self, activation=None, kernel_regularizer=None, name='AMPLayer', **kwargs):
        super(AMPLayer, self).__init__(name=name, **kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.mpl_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        node_feature_shape, _, edge_feature_shape, _ = input_shape
        self.wq = self.add_weight(
            shape=(
                node_feature_shape[-1], edge_feature_shape[-1]),
            trainable=True,
            name='wq'
        )
        self.wk = self.add_weight(
            shape=(
                edge_feature_shape[-1], edge_feature_shape[-1]),
            trainable=True,
            name='wk'
        )
        self.wv = self.add_weight(
            shape=(
                node_feature_shape[-1], node_feature_shape[-1]),
            trainable=True,
            name='wv'
        )


    def get_config(self):
        config = super(AMPLayer, self).get_config()
        config.update(
            {'activation': self.activation})
        return config

    def call(self, inputs):
        # node ->  N x node_feature
        # nlist -> N x NN
        # edges -> N x NN x edge_features
        # inv_degree -> N
        nodes, nlist, edges, inv_degree = inputs
        # Get node matrix sliced by nlist -> N x NN x node_features
        sliced_features = tf.gather(nodes, nlist)
        query = nodes @ self.wq
        keys = edges @ self.wk
        values = sliced_features @ self.wv
        qdot = tf.einsum('i,ijk,ik->ij', inv_degree, keys, query)
        b = tf.math.softmax(qdot, axis=-1)
        reduced = tf.einsum('ij,ijk->ik', b, values)
        out = self.activation(reduced)
        # output -> N x D number of atoms x node feature dimension
        if self.mpl_regularizer is not None:
            self.add_loss(self.mpl_regularizer(self.w))
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

    def __init__(self, low, high, count, name='rbf-layer', **kwargs):
        R'''
        :param low: lowest :math:`\mu`
        :type low: float
        :param high: high :math:`\mu` (inclusive)
        :type high: float
        :param count: Number of elements in :math:`\mu` and output last axis dimension
        :type count: int
        '''
        super(RBFExpansion, self).__init__(name=name, **kwargs)
        self.low = low
        self.high = high
        self.count = count

    def build(self, input_shape):
        self.centers = tf.cast(tf.linspace(
            self.low, self.high, self.count), dtype=tf.float32)
        self.gap = self.centers[1] - self.centers[0]

    def get_config(self):
        config = super(RBFExpansion, self).get_config()
        config.update(
            {'low': self.low, 'high': self.high, 'count': self.count})
        return config

    def call(self, inputs):
        rbf = tf.math.exp(-(inputs[..., tf.newaxis] -
                            self.centers)**2 / self.gap)
        return rbf
