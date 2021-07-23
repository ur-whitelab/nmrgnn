import numpy as np
import pickle
import os


import tensorflow as tf
import nmrdata
import nmrdata
import nmrgnn


def setup_optimizations():
    # tf.debugging.enable_check_numerics()

    tf.config.optimizer.set_jit(True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    #policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)


def load_baseline():
    from importlib_resources import files
    import nmrgnn.models
    fp = files(nmrgnn.models).joinpath(
        'baseline')
    return fp


def check_peaks(atoms, peaks, cutoff_sigma=4, warn_sigma=2.5):
    ''' Estimates if peaks are valid using known distribution of training chemical shifts. Returns
    True/False where True means likely valid and False means likely invalid.
    '''
    peak_standards = nmrdata.load_standards()
    confident = np.empty(atoms.shape[0], dtype=np.bool)
    confident[:] = True
    for i in range(len(atoms)):
        ps = peak_standards[int(np.nonzero(atoms[i])[0])]
        if ps[2] == 0 or (peaks[i] - ps[1])**2 / ps[2]**2 > warn_sigma**2:
            confident[i] = False
        # if ps[2] == 0 or (peaks[i] - ps[1])**2 / ps[2]**2 > cutoff_sigma**2:
        #    peaks[i] = np.nan
    # do we have at least 75% confidence?
    if np.sum(confident) / confident.shape[0] < 0.75:
        raise Warning(
            'Your peaks look awful. Likely solvent or missing hydrogens or bad units. Check README for suggestions')
    return confident


def load_data(tfrecords, validation, embeddings, scale=False):
    # load data and split into train/validation

    # need to load each tf record individually and split
    # so that we have equal representation in validation data
    data = None
    validation_data = None
    print(f'Loading from {len(tfrecords)} files')
    for tfr in tfrecords:
        d = nmrdata.dataset(tfr, embeddings=embeddings, label_info=True)
        # get size and split
        ds = len(list(d))
        vs = int(validation * ds)
        print(
            f'Loaded {tfr} and found {ds} records. Will keep {vs} for validation')
        v = d.take(vs)
        d = d.skip(vs)
        if data is None:
            data = d
            validation_data = v
        else:
            data = data.concatenate(d)
            validation_data = validation_data.concatenate(v)

    if scale:
        peak_standards = nmrdata.load_standards()
        peak_std = np.ones(100, dtype=np.float32)
        peak_avg = np.zeros(100, dtype=np.float32)
        for k, v in peak_standards.items():
            peak_std[k] = v[2]
            peak_avg[k] = v[1]

        train_data = data.map(
            lambda *x: unstandardize_labels(*x,
                                            peak_std=peak_std, peak_avg=peak_avg)
        )

    # shuffle train at each iteration
    train_data = data.shuffle(500, reshuffle_each_iteration=True)
    return train_data.prefetch(tf.data.experimental.AUTOTUNE), validation_data.cache()


def load_model(model_file=None):
    '''Load chemical shift prediction model. If no file given, the pre-trained model is loaded.
    '''
    setup_optimizations()

    if model_file is None:
        model_file = load_baseline()
    model_name = os.path.basename(model_file)

    model = tf.keras.models.load_model(
        model_file, custom_objects=nmrgnn.custom_objects)
    return model


def universe2graph(u, neighbor_number=16):
    '''Convert universe into tuple of graph objects. Universe is presumed to be in Angstrom. Universe should have explicit hydrogens

    Returns tuple with: atoms (one-hot element identity), nlist (neighbor list indices), edges (neighbor list distances), and inv_degree (inverse of degree for each atom).
    '''
    embeddings = nmrdata.load_embeddings()
    atoms, edges, nlist = nmrdata.parse_universe(
        u, neighbor_number, embeddings)
    mask = np.ones_like(atoms)
    inv_degree = tf.squeeze(tf.math.divide_no_nan(1.,
                                                  tf.reduce_sum(tf.cast(nlist > 0, tf.float32), axis=1)))
    return atoms, nlist, edges, inv_degree
