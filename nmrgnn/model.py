import tensorflow as tf
from tensorflow import keras
import numpy as np
import kerastuner as kt
import os
import nmrdata
from .layers import *
from .losses import *

# An object stores model hyper parameters


def build_GNNModel(hp=kt.HyperParameters()):
    hp.Int('atom_feature_size', 32, 256, step=32, default=256)
    hp.Int('edge_feature_size', 1, 16, step=4, default=4)
    hp.Int('edge_hidden_size', 16, 256, step=16, default=64)
    hp.Int('mp_layers', 1, 6, step=1, default=4)
    hp.Int('fc_layers', 2, 6, step=1, default=3)
    hp.Int('edge_fc_layers', 2, 6, step=1, default=2)
    hp.Choice('noise', [0.0, 0.01, 0.02, 0.05], ordered=True, default=0.02)
    hp.Fixed('rbf_low', 0.01)
    hp.Fixed('rbf_high', 0.12)
    hp.Choice('mp_activation', [
        'relu', 'softplus'])
    hp.Choice('fc_activation', [
        'relu', 'softplus'])

    # load peak standards
    standards = nmrdata.load_standards()

    model = GNNModel(hp, standards)

    # compile with MSLE (to treat vastly different label mags)
    optimizer = tf.keras.optimizers.Adam(
        hp.Choice('learning_rate', [5e-3, 1e-4, 1e-5], default=1e-4))
    loss = MeanSquaredLogartihmicErrorNames()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[tf.keras.metrics.MeanAbsoluteError()]
                  )
    return model


# Fully Connected Layers BLOCK for edge matrix


class EdgeFCBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(EdgeFCBlock, self).__init__(name='edge-fc-block')
        self.edge_fc = []
        # stack Dense Layers as a block
        for _ in range(hypers.get('edge_fc_layers') - 1):
            self.edge_fc.append(keras.layers.Dense(
                hypers.get('edge_hidden_size'), activation=hypers.get('fc_activation')))
        # activation function for the last layer is 'tanh'
        self.edge_fc.append(keras.layers.Dense(
            hypers.get('edge_feature_size'), activation="tanh"))

    def call(self, edge_input):
        # edge_input is untrained edge matrix
        x = self.edge_fc[0](edge_input)
        for i in range(1, len(self.edge_fc)):
            x = self.edge_fc[i](x)
        # output is trained edge matrix, same dimension as input
        return x


# Message Passing Layers BLOCK for node matrix
class MPBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(MPBlock, self).__init__(name='mp-block')
        self.mp = []
        # stack Message Passing Layers as a block
        for _ in range(hypers.get('mp_layers')):
            self.mp.append(MPLayer(hypers.get('mp_activation')))

    def call(self, inputs):
        # inputs should be in format
        # [nodes, nlist, edges, inv_degree]
        # where edges is the output from EdgeFCBlock
        # With Residue Block
        # Skip Connection applied at every layer
        nodes = inputs[0]
        for i in range(len(self.mp)):
            nodes = self.mp[i](inputs) + nodes
            # only nodes matrix is updated
            inputs[0] = nodes
        # Dimension of output nodes matrix should match input nodes matrix
        return nodes


# Fully Connected Layers BLOCK for nodes matrix, right after MPBlock
class FCBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(FCBlock, self).__init__(name='fc-block')
        self.fc = []
        # stack FC Layers as a block
        for _ in range(hypers.get('fc_layers') - 1):
            self.fc.append(keras.layers.Dense(
                hypers.get('atom_feature_size'), activation=hypers.get('fc_activation')))
        self.fc.append(keras.layers.Dense(
            hypers.get('atom_feature_size'), activation="tanh"))

    def call(self, nodes):
        for i in range(len(self.fc)):
            nodes = self.fc[i](nodes)
        # dimension of output should match with input
        return nodes


class GNNModel(keras.Model):
    def __init__(self, hypers, peak_standards):
        super(GNNModel, self).__init__(name='gnn-model')
        self.edge_rbf = RBFExpansion(
            hypers.get('rbf_low'), hypers.get('rbf_high'), hypers.get('edge_hidden_size'))
        self.edge_fc_block = EdgeFCBlock(hypers)
        self.mp_block = MPBlock(hypers)
        self.fc_block = FCBlock(hypers)
        self.noise_block = tf.keras.layers.GaussianNoise(hypers.get('noise'))
        self.hypers = hypers

        # we will use peak_standards now (large) and cut down later
        # This is because saving peak_standards is probelmatic
        LOTS_OF_ELEMENTS = 100
        # now expand to match range of peaks
        self.peak_std = np.ones(LOTS_OF_ELEMENTS, dtype=np.float32)
        self.peak_avg = np.zeros(LOTS_OF_ELEMENTS, dtype=np.float32)
        for k, v in peak_standards.items():
            self.peak_std[k] = v[2]
            self.peak_avg[k] = v[1]

    def build(self, input_shapes):
        # get number of elements
        num_elem = input_shapes[0][-1]
        self.out_layer = tf.keras.layers.Dense(num_elem)
        self.peak_std = self.peak_std[:num_elem]
        self.peak_avg = self.peak_avg[:num_elem]

    def call(self, inputs, training=None):
        # node_input should be 1 hot!
        # as written here, edge input is distance ONLY
        # modify if you want to include type informaton
        node_input, nlist_input, edge_input, inv_degree = inputs

        edge_mask = tf.cast(edge_input > 0, tf.float32)[..., tf.newaxis]

        noised_edges = self.noise_block(edge_input, training)
        rbf_edges = self.edge_rbf(noised_edges)
        # want to preserve zeros in input
        # so multiply here by mask (!)
        rbf_edges *= edge_mask
        edge_embeded = self.edge_fc_block(rbf_edges)
        # want to preserve zeros in input
        # so multiply here by mask (!)
        edge_embeded *= edge_mask
        mp_inputs = [node_input, nlist_input, edge_embeded, inv_degree]
        semi_nodes = self.mp_block(mp_inputs)
        out_nodes = self.fc_block(semi_nodes)
        full_peaks = self.out_layer(out_nodes)

        peaks = tf.reduce_sum(full_peaks * node_input * self.peak_std +
                              node_input * self.peak_avg, axis=-1)
        return peaks
