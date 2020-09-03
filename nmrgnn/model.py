import tensorflow as tf
from tensorflow import keras
import os
from .layers import *

# An object stores model hyper parameters
class GNNHypers:
    def __init__(self):
        self.ATOM_FEATURE_SIZE = 256 # Size of space ito which we project elements
        self.EDGE_FEATURE_SIZE = 4 # Size of space onto which we project bonds (singlw, double, etc.)
        self.MP_LAYERS = 4 # Number of layers in Message Passing
        self.FC_LAYERS = 3 # Number of layers in Fully Connected
        self.EDGE_FC_LAYERS = 2 # Number of layers in Edge FC (Edge embedding)
        self.MP_ACTIVATION = tf.keras.activations.relu
        self.FC_ACTIVATION = tf.keras.activations.relu

# Fully Connected Layers BLOCK for edge matrix
class EdgeFCBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(EdgeFCBlock, self).__init__()
        self.hypers = hypers
        self.edge_fc = []
        # stack Dense Layers as a block
        for _ in range(self.hypers.EDGE_FC_LAYERS - 1):
            self.edge_fc.append(keras.layers.Dense(hypers.EDGE_FEATURE_SIZE, activation=hypers.FC_ACTIVATION)) 
        self.edge_fc.append(keras.layers.Dense(hypers.EDGE_FEATURE_SIZE, activation="tanh")) # activation function for the last layer is 'tanh'

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
        # [nodes, nlist, edges]
        # where edges is the output from EdgeFCBlock
        nodes, nlist, edges = inputs
        # With Residue Block
        # Skip Connection applied at every layer
        for i in range(self.hypers.MP_LAYERS):
            nodes = self.mp[i](inputs) + nodes
            # only nodes matrix is updated
            inputs = [nodes, nlist, edges]
        # Dimension of output nodes matrix shoulds matches input nodes matrix
        return nodes    


# Fully Connected Layers BLOCK for nodes matrix, right after MPBlock
class FCBlock(keras.layers.Layer):
    def __init__(self, hypers):
        super(FCBlock, self).__init__()
        self.hypers = hypers
        self.fc = []
        # stack FC Layers as a block
        for _ in range(self.hypers.FC_LAYERS - 1):
            self.fc.append(keras.layers.Dense(self.hypers.ATOM_FEATURE_SIZE, activation=self.hypers.FC_ACTIVATION))
        self.fc.append(keras.layers.Dense(self.hypers.ATOM_FEATURE_SIZE, activation="tanh"))
    
    def call(self, nodes):
        for i in range(self.hypers.FC_LAYERS):
            nodes = self.fc[i](nodes)
        # dimension of output should match with input
        return nodes


class GNNModel(keras.Model):
    def __init__(self, hypers):
        super(GNNModel, self).__init__()
        self.hypers = hypers
        self.edge_fc_block = EdgeFCBlock(hypers)
        self.mp_block = MPBlock(hypers)
        self.fc_block = FCBlock(hypers)

    def call(self, inputs):
        node_input, nlist_input, edge_input = inputs
        edge_embeded = self.edge_fc_block(edge_input)
        mp_inputs = [node_input, nlist_input, edge_embeded]
        semi_nodes = self.mp_block(mp_inputs)
        out_nodes = self.fc_block(semi_nodes)

        return out_nodes





