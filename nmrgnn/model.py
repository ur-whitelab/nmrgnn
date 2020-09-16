import tensorflow as tf
from tensorflow import keras
import numpy as np
import kerastuner as kt
import os
import nmrdata
from .layers import *
from .losses import *
from .metrics import *

# An object stores model hyper parameters


def build_GNNModel(hp=kt.HyperParameters(), metrics=True):
    hp.Choice('atom_feature_size', [32, 64, 128, 256], ordered=True, default=64)
    hp.Choice('edge_feature_size', [1, 2, 4, 8, 16, 32], ordered=True, default=16)
    hp.Choice('edge_hidden_size', [16, 32, 64, 128, 256], ordered=True, default=128)
    hp.Int('mp_layers', 1, 6, step=1, default=4)
    hp.Int('fc_layers', 2, 6, step=1, default=3)
    hp.Int('edge_fc_layers', 2, 6, step=1, default=4)
    hp.Choice('noise', [0.0, 0.005, 0.01, 0.02, 0.05], ordered=True, default=0.02)
    hp.Fixed('rbf_low', 0.005)
    hp.Fixed('rbf_high', 0.15)
    hp.Choice('mp_activation', [
        'relu', 'softplus'], default='relu')
    hp.Choice('fc_activation', [
        'relu', 'softplus'], default='relu')

    # load peak standards
    standards = nmrdata.load_standards()

    model = GNNModel(hp, standards)

    # compile with MSLE (to treat vastly different label mags)
    optimizer = tf.keras.optimizers.Adam(
        hp.Choice('learning_rate', [1e-2, 1e-3, 5e-3, 1e-4, 1e-5], default=1e-4))
    loss = corr_loss
    embeddings = nmrdata.load_embeddings()
    label_idx = type_mask(r'.*\-H.*', embeddings, regex=True)
    h_mae = NameMAE(label_idx, name='h_mae')
    label_idx = type_mask(r'.*\-N.*', embeddings, regex=True)
    n_mae = NameMAE(label_idx, name='n_mae')
    label_idx = type_mask(r'.*\-C.*', embeddings, regex=True)
    c_mae = NameMAE(label_idx, name='c_mae')
    label_idx = type_mask(r'.*\-H', embeddings, regex=True)
    hn_mae = NameMAE(label_idx, name='hn_mae')
    label_idx = type_mask(r'.*\-HA*', embeddings, regex=True)
    ha_mae = NameMAE(label_idx, name='ha_mae')
    label_idx = type_mask(r'.*\-H.*', embeddings, regex=True)
    h_r2 = NameR2(label_idx, name='h_r2')
    label_idx = type_mask(r'.*\-N.*', embeddings, regex=True)
    n_r2 = NameR2(label_idx, name='n_r2')
    label_idx = type_mask(r'.*\-C.*', embeddings, regex=True)
    c_r2 = NameR2(label_idx, name='c_r2')
    label_idx = type_mask(r'.*\-H', embeddings, regex=True)
    hn_r2 = NameR2(label_idx, name='hn_r2')
    label_idx = type_mask(r'.*\-HA*', embeddings, regex=True)
    ha_r2 = NameR2(label_idx, name='ha_r2')
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[
                      h_mae,
                      n_mae,
                      c_mae,
                      hn_mae,
                      ha_mae,
                      h_r2,
                      n_r2,
                      c_r2,
                      hn_r2,
                      ha_r2
                  ] if metrics else None
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
        self.hypers = hypers

    def call(self, edge_input):
        # edge_input is untrained edge matrix
        x = self.edge_fc[0](edge_input)
        for i in range(1, len(self.edge_fc)):
            x = self.edge_fc[i](x)
        # output is trained edge matrix, same dimension as input
        return x

    def get_config(self):
        config = super(EdgeFCBlock, self).get_config()
        config.update(
            {'hypers': self.hypers})
        return config


# Message Passing Layers BLOCK for node matrix
class MPBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(MPBlock, self).__init__(name='mp-block')
        self.mp = []
        # stack Message Passing Layers as a block
        for _ in range(hypers.get('mp_layers')):
            self.mp.append(MPLayer(hypers.get('mp_activation')))
        self.hypers = hypers

    def call(self, inputs):
        # inputs should be in format
        # [nodes, nlist, edges, inv_degree]
        # where edges is the output from EdgeFCBlock
        # With Residue Block
        # Skip Connection applied at every layer
        nodes = inputs[0]
        for i in range(len(self.mp)):
            # only nodes matrix is updated
            nodes = self.mp[i]([nodes] + inputs[1:]) + nodes
        # Dimension of output nodes matrix should match input nodes matrix
        return nodes

    def get_config(self):
        config = super(MPBlock, self).get_config()
        config.update(
            {'hypers': self.hypers})
        return config


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
        self.hypers = hypers

    def call(self, nodes):
        for i in range(len(self.fc)):
            nodes = self.fc[i](nodes)
        # dimension of output should match with input
        return nodes

    def get_config(self):
        config = super(FCBlock, self).get_config()
        config.update(
            {'hypers': self.hypers})
        return config


class GNNModel(keras.Model):
    def __init__(self, hypers, peak_standards, name='gnn-model', **kwargs):
        super(GNNModel, self).__init__(name=name, **kwargs)
        self.edge_rbf = RBFExpansion(
            hypers.get('rbf_low'), hypers.get('rbf_high'), hypers.get('edge_hidden_size'))
        self.edge_fc_block = EdgeFCBlock(hypers)
        self.mp_block = MPBlock(hypers)
        self.fc_block = FCBlock(hypers)
        self.noise_block = tf.keras.layers.GaussianNoise(hypers.get('noise'))
        self.hypers = hypers
        self.embed_dim = hypers.get('atom_feature_size')
        # we will use peak_standards now (large) and cut down later
        # This is because saving peak_standards is probelmatic
        LOTS_OF_ELEMENTS = 100
        # now expand to match range of peaks
        self.peak_std = np.ones(LOTS_OF_ELEMENTS, dtype=np.float32)
        self.peak_avg = np.zeros(LOTS_OF_ELEMENTS, dtype=np.float32)
        for k, v in peak_standards.items():
            self.peak_std[k] = v[2]
            self.peak_avg[k] = v[1]

    def get_config(self):
        config = super(GNNModel, self).get_config()
        config.update(
            {'hypers': self.hypers, 'peak_standards': self.peak_standards})
        return config

    def build(self, input_shapes):
        # get number of elements
        num_elem = input_shapes[0][-1]
        self.out_layer = tf.keras.layers.Dense(num_elem)
        # they are already one-hots, so no need to use proper embedding
        self.embed_layer = tf.keras.layers.Dense(self.embed_dim, use_bias = False)
        self.peak_std = self.peak_std[:num_elem]
        self.peak_avg = self.peak_avg[:num_elem]

    def call(self, inputs, training=False):
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
        node_embed = self.embed_layer(node_input)
        mp_inputs = [node_embed, nlist_input, edge_embeded, inv_degree]
        semi_nodes = self.mp_block(mp_inputs)
        out_nodes = self.fc_block(semi_nodes)
        full_peaks = self.out_layer(out_nodes)
        #if training:
        #    peaks = tf.reduce_sum(full_peaks * node_input, axis=-1)
        #else:
        peaks = tf.reduce_sum(full_peaks * node_input * self.peak_std +
                              node_input * self.peak_avg, axis=-1)
        return peaks
