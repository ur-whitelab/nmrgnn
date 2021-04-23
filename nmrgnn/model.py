import tensorflow as tf
from tensorflow import keras
import numpy as np
import kerastuner as kt
import os
import nmrdata
from .layers import *
from .losses import *
from .metrics import *


def build_GNNModel(hp=kt.HyperParameters(), metrics=True, loss_balance=1.0):
    '''Build model with hyper parameter object'''
    # small AMP (170k parameters)
    #hp.Choice('atom_feature_size', [32, 64, 128, 256], ordered=True, default=128)
    #hp.Choice('edge_feature_size', [1, 2, 3, 64], ordered=True, default=64)
    #hp.Choice('edge_hidden_size', [16, 32, 64, 128, 256], ordered=True, default=64)
    #hp.Int('mp_layers', 1, 6, step=1, default=4)
    #hp.Int('fc_layers', 2, 6, step=1, default=3)
    #hp.Int('edge_fc_layers', 2, 6, step=1, default=3)

    hp.Choice('atom_feature_size', [32, 64, 128, 256], ordered=True, default=256)
    hp.Choice('edge_feature_size', [1, 2, 3, 8, 64], ordered=True, default=3)
    hp.Choice('edge_hidden_size', [16, 32, 64, 128, 256], ordered=True, default=128)
    hp.Int('mp_layers', 1, 6, step=1, default=4)
    hp.Int('fc_layers', 2, 6, step=1, default=4)
    hp.Int('edge_fc_layers', 2, 6, step=1, default=4)

    hp.Choice('noise', [0.0, 0.025, 0.05, 0.1], ordered=True, default=0.025)
    hp.Choice('dropout', [True, False], default=True)
    hp.Fixed('rbf_low', 0.005)
    hp.Fixed('rbf_high', 0.20)
    hp.Choice('mp_activation', [
        'relu', 'softplus', 'tanh'], default='softplus')
    hp.Choice('fc_activation', [
        'relu', 'softplus'], default='softplus')

    # load peak standards
    standards = nmrdata.load_standards()

    model = GNNModel(hp, standards)

    # compile with MSLE (to treat vastly different label mags)
    optimizer = tf.keras.optimizers.Adam(
        hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4, 1e-5], default=1e-4))

    embeddings = nmrdata.load_embeddings()

    #label_idx = type_mask(r'.*\-H.*', embeddings, regex=True)
    label_idx = type_mask(r'.*', embeddings, regex=True)
    corr_loss = NameLoss(label_idx, s=loss_balance)
    loss = corr_loss



    label_idx = type_mask(r'.*\-H.*', embeddings, regex=True)
    h_rmsd = NameRMSD(label_idx, name='h_rmsd')
    label_idx = type_mask(r'.*\-N.*', embeddings, regex=True)
    n_rmsd = NameRMSD(label_idx, name='n_rmsd')
    label_idx = type_mask(r'.*\-C.*', embeddings, regex=True)
    c_rmsd = NameRMSD(label_idx, name='c_rmsd')
    label_idx = type_mask(r'.*\-H$', embeddings, regex=True)
    hn_rmsd = NameRMSD(label_idx, name='hn_rmsd')
    label_idx = type_mask(r'.*\-HA*', embeddings, regex=True)
    ha_rmsd = NameRMSD(label_idx, name='ha_rmsd')
    label_idx = type_mask(r'.*\-H.*', embeddings, regex=True)
    h_r = NameCorr(label_idx, name='h_r')
    label_idx = type_mask(r'.*\-N.*', embeddings, regex=True)
    n_r = NameCorr(label_idx, name='n_r')
    label_idx = type_mask(r'.*\-C.*', embeddings, regex=True)
    c_r = NameCorr(label_idx, name='c_r')
    label_idx = type_mask(r'.*\-H$', embeddings, regex=True)
    hn_r = NameCorr(label_idx, name='hn_r')
    label_idx = type_mask(r'.*\-HA.*', embeddings, regex=True)
    ha_r = NameCorr(label_idx, name='ha_r')    
    label_idx = type_mask(r'.*\-HA.*', embeddings, regex=True)
    ha_r = NameCorr(label_idx, name='ha_r')    
    ha_count = NameCount(label_idx, name='avg_ha_count')

    label_idx = type_mask(r'DFT.*', embeddings, regex=True)
    dft_r = NameCorr(label_idx, name='dft_r')    
    dft_count = NameCount(label_idx, name='avg_dft_count')
    label_idx = type_mask(r'MB.*', embeddings, regex=True)
    mb_r = NameCorr(label_idx, name='mb_r')    
    mb_count = NameCount(label_idx, name='avg_mb_count')


    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[
                      h_rmsd,
                      n_rmsd,
                      c_rmsd,
                      hn_rmsd,
                      ha_rmsd,
                      h_r,
                      n_r,
                      c_r,
                      hn_r,
                      ha_r, ha_count,
                      mb_r, mb_count,
                      dft_r, dft_count             
                  ] if metrics else None
                  )
    return model


# Fully Connected Layers BLOCK for edge matrix

class EdgeFCBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(EdgeFCBlock, self).__init__(name='edge-fc-block')
        
        # add l1 regularizer to 
        # input, so that we 
        # zero-out unused distance features
        # off until I know I need it
        self.edge_fc = [keras.layers.Dense(
            hypers.get('edge_hidden_size'), 
            activation=hypers.get('fc_activation'),
            #kernel_regularizer='l1',
        )]
        # stack Dense Layers as a block
        for _ in range(hypers.get('edge_fc_layers') - 2):
            self.edge_fc.append(keras.layers.Dense(
                hypers.get('edge_hidden_size'), activation=hypers.get('fc_activation')))
        self.edge_fc.append(keras.layers.Dense(
            hypers.get('edge_feature_size')))

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
            hypers.get('atom_feature_size') // 2, activation=hypers.get('fc_activation')))
        self.hypers = hypers

    def call(self, nodes):
        for i in range(len(self.fc) - 1):
            nodes = self.fc[i](nodes) + nodes
        nodes = self.fc[-1](nodes)
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
        if hypers.get('dropout'):
            self.dropout = tf.keras.layers.Dropout(0.2)
        else:
            self.dropout = None
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
        if self.dropout is not None:
            out_nodes = self.dropout(out_nodes, training)
        full_peaks = self.out_layer(out_nodes)
        #if training:
        #    peaks = tf.reduce_sum(full_peaks * node_input, axis=-1)
        #else:            
        peaks = tf.reduce_sum(full_peaks * node_input * self.peak_std +
                              node_input * self.peak_avg, axis=-1)
        return peaks
