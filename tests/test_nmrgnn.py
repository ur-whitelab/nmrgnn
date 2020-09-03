import tensorflow as tf
import unittest
import nmrgnn
import numpy as np



class test_imports(unittest.TestCase):
    def test_importable(self):
        pass


class test_mpl(unittest.TestCase):

    def test_mpl_build(self):
        # Nodes ->  5 of them, each with features 16
        nodes = tf.ones((5, 16))
        # neighbors ->  5 nodes, 2 neighbors
        nlist = np.zeros((5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for i in range(5):
            for k, j in enumerate(range(-1, 3, 2)):
                nlist[i, k] = (i + j) % 5
        nlist = tf.constant(nlist)
        # 5 nodes, 2 neighbors, 2 feature
        edges = tf.ones((5, 2, 2))
        mpl = nmrgnn.MPLayer()
        new_nodes = mpl([nodes, nlist, edges])
        assert new_nodes.shape == nodes.shape

    def test_mpl_call(self):
        pass


class test_gnnhypers(unittest.TestCase):

    def test_gnnhypers_setup(self):
        pass


class test_edge_fc_block(unittest.TestCase):

    def test_edgeFCBlock_call(self):
        edge_input = tf.ones((5, 2, 2))
        edgeFCBlock = nmrgnn.EdgeFCBlock(nmrgnn.GNNHypers())
        edge_output = edgeFCBlock(edge_input)
        assert edge_output.shape[-1] == nmrgnn.GNNHypers().EDGE_FEATURE_SIZE
        assert edge_output.shape[:-1] == edge_input.shape[:-1]



class test_mp_block(unittest.TestCase):

    def test_mpBlock_call(self):
        nodes = tf.ones((5, 16))
        # neighbors ->  5 nodes, 2 neighbors
        nlist = np.zeros((5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for i in range(5):
            for k, j in enumerate(range(-1, 3, 2)):
                nlist[i, k] = (i + j) % 5
        nlist = tf.constant(nlist)
        # batch size 2, 5 nodes, 2 neighbors, 4 feature
        edges = tf.ones((5, 2, 2))
        mp_block = nmrgnn.MPBlock(nmrgnn.GNNHypers()) # shapes are specified inside the block
        out_nodes = mp_block([nodes, nlist, edges])
        assert out_nodes.shape == nodes.shape



class test_fc_block(unittest.TestCase):

    def test_fcBlock_call(self):
        nodes = tf.ones((5, 16))
        fcBlock = nmrgnn.FCBlock(nmrgnn.GNNHypers())
        new_nodes = fcBlock(nodes)
        assert new_nodes.shape[-1] == nmrgnn.GNNHypers().ATOM_FEATURE_SIZE
        assert new_nodes.shape[:-1] == nodes.shape[:-1]

class test_gnnmodel(unittest.TestCase):

    def test_gnnmodel_build(self):
        nodes = tf.ones((5, 16))
        # neighbors -> 5 nodes, 2 neighbors
        nlist = np.zeros((5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for i in range(5):
            for k, j in enumerate(range(-1, 3, 2)):
                nlist[i, k] = (i + j) % 5
        nlist = tf.constant(nlist)
        # 5 nodes, 2 neighbors, 4 feature
        edges = tf.ones((5, 2, 2))
        inputs = [nodes, nlist, edges]
        model = nmrgnn.GNNModel(nmrgnn.GNNHypers())
        out_nodes = model(inputs)
        assert out_nodes.shape[-1] == nmrgnn.GNNHypers().ATOM_FEATURE_SIZE
        assert out_nodes.shape[:-1] == nodes.shape[:-1]
            


