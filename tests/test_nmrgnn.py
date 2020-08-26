import tensorflow as tf
import unittest
import nmrgnn


class test_imports(unittest.TestCase):
    def test_importable(self):
        pass


class test_mpl(unittest.TestCase):

    def test_mpl_build(self):
        # Nodes -> batch size 2, 5 of them, each with features 3
        nodes = tf.ones((2, 5, 3))
        # neighbors -> batch size 2, 5 nodes, 2 neighbors
        nlist = np.zeros((2, 5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for b in range(2):
            for i in range(5):
                for k, j in enumerate(range(-1, 3, 2)):
                    nlist[b, i, k] = (i + j) % 5

        nlist = tf.constant(nlist)
        # batch size 2, 5 nodes, 2 neighbors, 1 feature
        edges = tf.ones((2, 5, 2, 1))

        mpl = nmrgnn.MPLayer(name='mpl')
        new_nodes = mpl([nodes, nlist, edges])
        self.assert(new_nodes.shape == nodes.shape)
