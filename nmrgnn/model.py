import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from .layers import *

# An object stores model hyper parameters


class GNNHypers:
    def __init__(self):
        self.ATOM_FEATURE_SIZE = 256  # Size of space ito which we project elements
        # Size of space onto which we project bonds (singlw, double, etc.)
        self.EDGE_FEATURE_SIZE = 4
        self.MP_LAYERS = 4  # Number of layers in Message Passing
        self.FC_LAYERS = 3  # Number of layers in Fully Connected
        self.EDGE_FC_LAYERS = 2  # Number of layers in Edge FC (Edge embedding)
        self.MP_ACTIVATION = tf.keras.activations.relu
        self.FC_ACTIVATION = tf.keras.activations.relu
        self.RBF_HIGH = 0.12  # nm
        self.RBF_LOW = 0.05

# Fully Connected Layers BLOCK for edge matrix


class EdgeFCBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(EdgeFCBlock, self).__init__()
        self.hypers = hypers
        self.edge_fc = []
        # stack Dense Layers as a block
        for _ in range(self.hypers.EDGE_FC_LAYERS - 1):
            self.edge_fc.append(keras.layers.Dense(
                hypers.EDGE_FEATURE_SIZE, activation=hypers.FC_ACTIVATION))
        # activation function for the last layer is 'tanh'
        self.edge_fc.append(keras.layers.Dense(
            hypers.EDGE_FEATURE_SIZE, activation="tanh"))

    def call(self, edge_input):
        # edge_input is untrained edge matrix
        x = self.edge_fc[0](edge_input)
        for i in range(1, self.hypers.EDGE_FC_LAYERS):
            x = self.edge_fc[i](x)
        # output is trained edge matrix, same dimension as input
        return x


# Message Passing Layers BLOCK for node matrix
class MPBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(MPBlock, self).__init__()
        self.hypers = hypers
        self.mp = []
        # stack Message Passing Layers as a block
        for _ in range(self.hypers.MP_LAYERS):
            self.mp.append(MPLayer(hypers.MP_ACTIVATION))

    def call(self, inputs):
        # inputs should be in format
        # [nodes, nlist, edges, inv_degree]
        # where edges is the output from EdgeFCBlock
        # With Residue Block
        # Skip Connection applied at every layer
        nodes = inputs[0]
        for i in range(self.hypers.MP_LAYERS):
            nodes = self.mp[i](inputs) + nodes
            # only nodes matrix is updated
            inputs[0] = nodes
        # Dimension of output nodes matrix should match input nodes matrix
        return nodes


# Fully Connected Layers BLOCK for nodes matrix, right after MPBlock
class FCBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(FCBlock, self).__init__()
        self.hypers = hypers
        self.fc = []
        # stack FC Layers as a block
        for _ in range(self.hypers.FC_LAYERS - 1):
            self.fc.append(keras.layers.Dense(
                self.hypers.ATOM_FEATURE_SIZE, activation=self.hypers.FC_ACTIVATION))
        self.fc.append(keras.layers.Dense(
            self.hypers.ATOM_FEATURE_SIZE, activation="tanh"))

    def call(self, nodes):
        for i in range(self.hypers.FC_LAYERS):
            nodes = self.fc[i](nodes)
        # dimension of output should match with input
        return nodes


class GNNModel(keras.Model):
    def __init__(self, hypers, peak_standards):
        super(GNNModel, self).__init__()
        self.hypers = hypers
        self.edge_rbf = RBFExpansion(
            hypers.RBF_LOW, hypers.RBF_HIGH, hypers.EDGE_FEATURE_SIZE)
        self.edge_fc_block = EdgeFCBlock(hypers)
        self.mp_block = MPBlock(hypers)
        self.fc_block = FCBlock(hypers)
        self.peak_standards = peak_standards

    def build(self, input_shapes):
        # get number of elements
        num_elem = input_shapes[0][-1]
        self.out_layer = tf.keras.layers.Dense(num_elem)

        # now expand to match range of peaks
        self.peak_std = np.ones(num_elem, dtype=np.float32)
        self.peak_avg = np.zeros(num_elem, dtype=np.float32)
        for k, v in self.peak_standards.items():
            self.peak_std[k] = v[2]
            self.peak_avg[k] = v[1]

    def call(self, inputs, training=None):
        # node_input should be 1 hot!
        node_input, nlist_input, edge_input, inv_degree = inputs

        rbf_edges = self.edge_rbf(edge_input)
        edge_embeded = self.edge_fc_block(rbf_edges)
        mp_inputs = [node_input, nlist_input, edge_embeded, inv_degree]
        semi_nodes = self.mp_block(mp_inputs)
        out_nodes = self.fc_block(semi_nodes)
        full_peaks = self.out_layer(out_nodes)

        peaks = tf.reduce_sum(full_peaks * node_input * self.peak_std +
                              node_input * self.peak_avg, axis=-1)
        return peaks
