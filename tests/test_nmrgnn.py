import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
import unittest
import nmrgnn
import numpy as np
import shutil
import os


class TestImports(unittest.TestCase):
    def test_importable(self):
        pass


class TestMPL(unittest.TestCase):

    def test_mpl_build(self):
        # Nodes ->  5 of them, each with features 16
        nodes = tf.one_hot([2, 4, 0, 1, 3], 16)
        # neighbors ->  5 nodes, 2 neighbors
        nlist = np.zeros((5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for i in range(5):
            for k, j in enumerate(range(-1, 3, 2)):
                nlist[i, k] = (i + j) % 5
        nlist = tf.constant(nlist)
        # 5 nodes, 2 neighbors, 2 feature
        edges = tf.ones((5, 2, 2))
        # make inv degree (each one has two connections)
        inv_degree = np.ones((5,)) / 2
        mpl = nmrgnn.MPLayer()
        new_nodes = mpl([nodes, nlist, edges, inv_degree])
        assert new_nodes.shape == nodes.shape

    def test_ampl_build(self):
        # Nodes ->  5 of them, each with features 16
        nodes = tf.one_hot([2, 4, 0, 1, 3], 16)
        # neighbors ->  5 nodes, 2 neighbors
        nlist = np.zeros((5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for i in range(5):
            for k, j in enumerate(range(-1, 3, 2)):
                nlist[i, k] = (i + j) % 5
        nlist = tf.constant(nlist)
        # 5 nodes, 2 neighbors, 2 feature
        edges = tf.ones((5, 2, 2))
        # make inv degree (each one has two connections)
        inv_degree = np.ones((5,)) / 2
        mpl = nmrgnn.AMPLayer()
        new_nodes = mpl([nodes, nlist, edges, inv_degree])
        assert new_nodes.shape == nodes.shape

    def test_mpl_call(self):
        pass


class TestGNNHypers(unittest.TestCase):

    def test_gnnhypers_setup(self):
        pass


class TestEdgeFCBlock(unittest.TestCase):

    def test_edgeFCBlock_call(self):
        model = nmrgnn.build_GNNModel()
        hypers = model.hypers
        edge_input = tf.ones((5, 2, 2))
        edgeFCBlock = nmrgnn.EdgeFCBlock(hypers)
        edge_output = edgeFCBlock(edge_input)
        assert edge_output.shape[-1] == hypers.get('edge_feature_size')
        assert edge_output.shape[:-1] == edge_input.shape[:-1]


class TestMPBlock(unittest.TestCase):

    def test_mpBlock_call(self):
        model = nmrgnn.build_GNNModel()
        hypers = model.hypers
        nodes = tf.one_hot([2, 0, 1, 3, 3], 16)
        # neighbors ->  5 nodes, 2 neighbors
        nlist = np.zeros((5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for i in range(5):
            for k, j in enumerate(range(-1, 3, 2)):
                nlist[i, k] = (i + j) % 5
        nlist = tf.constant(nlist)
        # 5 nodes, 2 neighbors, 4 feature
        edges = tf.ones((5, 2, 2))
        # make inv degree (each one has two connections)
        inv_degree = np.ones((5,)) / 2
        # shapes are specified inside the block
        mp_block = nmrgnn.MPBlock(hypers)
        out_nodes = mp_block([nodes, nlist, edges, inv_degree])
        assert out_nodes.shape == nodes.shape


class TestFCBlock(unittest.TestCase):

    def test_fcBlock_call(self):
        model = nmrgnn.build_GNNModel()
        hypers = model.hypers
        nodes = tf.ones((5, hypers.get('atom_feature_size')))
        fcBlock = nmrgnn.FCBlock(hypers)
        new_nodes = fcBlock(nodes)
        assert new_nodes.shape[-1] == hypers.get('atom_feature_size') // 2
        assert new_nodes.shape[-1] == nodes.shape[-1] // 2


class TestMetrics(unittest.TestCase):

    def test_name_rmsd(self):
        embeddings = {'name': {'ALA-N': 4, 'GLU-N': 2, 'GLU-H': 3}}
        label_idx = nmrgnn.type_mask(r'.*\-H', embeddings, regex=True)
        nm = nmrgnn.NameRMSD(label_idx)
        y = (tf.zeros((5,)), tf.constant([4., 3, 3, 2, 4]), tf.ones(5,))
        y = tf.stack([*y], axis=1)
        y_pred = np.zeros((5,))
        y_pred[1] = 5
        nm.update_state(y, y_pred)
        np.testing.assert_allclose(nm.result(), np.sqrt(5.0**2 / 2))

        label_idx = nmrgnn.type_mask(r'GLU-H', embeddings, regex=True)
        nm = nmrgnn.NameRMSD(label_idx)
        nm.update_state(y, y_pred)
        np.testing.assert_allclose(nm.result(), np.sqrt(5.0**2 / 2))

        y_pred[:] = 0
        y_pred[-2] = 5
        label_idx = nmrgnn.type_mask(r'GLU\-.*', embeddings, regex=True)
        nm = nmrgnn.NameRMSD(label_idx)
        nm.update_state(y, y_pred)
        np.testing.assert_allclose(nm.result(), np.sqrt(5**2 / 3))

        with self.assertRaises(ValueError):
            nm = nmrgnn.type_mask(r'LYS\-.*', embeddings, regex=True)


class TestSerialize(unittest.TestCase):
    def test_mpl(self):
        mpl = nmrgnn.MPLayer()
        serialized_layer = tf.keras.layers.serialize(mpl)
        new_layer = tf.keras.layers.deserialize(
            serialized_layer, custom_objects=nmrgnn.custom_objects
        )

    def test_rbf(self):
        layer = nmrgnn.RBFExpansion(0, 10, 100)
        serialized_layer = tf.keras.layers.serialize(layer)
        new_layer = tf.keras.layers.deserialize(
            serialized_layer, custom_objects=nmrgnn.custom_objects
        )

    def test_model(self):
        nodes = tf.one_hot([2, 4, 1, 3, 3], 16)
        # neighbors -> 5 nodes, 2 neighbors
        nlist = np.zeros((5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for i in range(5):
            for k, j in enumerate(range(-1, 3, 2)):
                nlist[i, k] = (i + j) % 5
        nlist = tf.constant(nlist)
        # 5 nodes, 2 neighbors, 4 feature
        edges = tf.ones((5, 2))
        # make inv degree (each one has two connections)
        inv_degree = np.ones((5,)) / 2
        inputs = [nodes, nlist, edges, inv_degree]

        # make peak standards
        ps = {}
        for i in range(16):
            ps[i] = ('F', 0, 1)

        model = nmrgnn.build_GNNModel()
        out_nodes = model(inputs)
        try:
            model.save('gnn_model_load_test')
            del model
            tf.keras.models.load_model(
                'gnn_model_load_test', custom_objects=nmrgnn.custom_objects)
        finally:
            shutil.rmtree('gnn_model_load_test')


class TestGNNModel(unittest.TestCase):

    def test_loss(self):
        y = (tf.zeros((5,)), tf.constant([4., 3, 3, 2, 4]))
        y = tf.stack([*y], axis=1)
        y_true = tf.ones(5,)
        embeddings = {'name': {'ALA-N': 4, 'GLU-N': 2, 'GLU-H': 3}}
        label_idx = nmrgnn.type_mask(r'.*\-H', embeddings, regex=True)
        loss = nmrgnn.NameLoss(label_idx, s=0.5)
        loss(y, y_true)

    def test_gnnmodel_build(self):
        nodes = tf.one_hot([2, 4, 1, 3, 3], 16)
        # neighbors -> 5 nodes, 2 neighbors
        nlist = np.zeros((5, 2), dtype=np.int)
        # make each atom neighbors with subsequent 2 atoms, mod 5
        for i in range(5):
            for k, j in enumerate(range(-1, 3, 2)):
                nlist[i, k] = (i + j) % 5
        nlist = tf.constant(nlist)
        # 5 nodes, 2 neighbors, 4 feature
        edges = tf.ones((5, 2))
        # make inv degree (each one has two connections)
        inv_degree = np.ones((5,)) / 2
        inputs = [nodes, nlist, edges, inv_degree]

        # make peak standards
        ps = {}
        for i in range(16):
            ps[i] = ('F', 0, 1)

        model = nmrgnn.build_GNNModel()
        out_nodes = model(inputs)
        # one peak per atom
        assert out_nodes.shape == (nodes.shape[0],)

        out_nodes = model(inputs)
        assert out_nodes.shape == (nodes.shape[0],)


class TestLibrary(unittest.TestCase):
    def test_load_model(self):
        model = nmrgnn.load_model()

    def test_universe2graph(self):
        import MDAnalysis as md
        u = md.Universe(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '108M.pdb'))
        x = nmrgnn.universe2graph(u)

    def test_check_peaks(self):
        model = nmrgnn.load_model()
        import MDAnalysis as md
        u = md.Universe(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '108M.pdb'))
        x = nmrgnn.universe2graph(u)
        peaks = model(x)
        nmrgnn.check_peaks(x[0], peaks)

    def test_traj_analysis(self):
        model = nmrgnn.load_model()
        import MDAnalysis as md
        u = md.Universe(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '7lgi.pdb.gz'))
        p1 = None
        for ts in u.trajectory:
            x = nmrgnn.universe2graph(u)
            peaks = model(x)
            if p1 is None:
                p1 = peaks
            nmrgnn.check_peaks(x[0], peaks)
        assert np.mean((peaks - p1)**2) > 1
