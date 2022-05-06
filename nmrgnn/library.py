import numpy as np
import pickle
import os


import tensorflow as tf
import nmrdata
import nmrdata
import kdens
import nmrgnn


def setup_optimizations():
    # tf.debugging.enable_check_numerics()

    tf.config.optimizer.set_jit(True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    # policy = mixed_precision.Policy('mixed_float16')
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


def squash_batch(state, x, L):
    '''Squash batch of tensors into one big tensor.

    The tensors come from sampling from datasets.
    If original shape of tensor is (N_0, F), (N_1, F)
    and batch size (L) is 2, they will be combined
    with a few to make (N_0 + N_1, F)

    L - number of elements in batch
    x - current record
    '''
    # Have B batches of tuples
    # Want to squash on leading dimension

    # s -> current sample
    # i -> index
    # n -> total items of current batch
    s, i, n = state
    if i % L == 0:
        s = x
        n = tf.shape(x[0][0])[0]
    else:
        snew = [[tf.concat([s[0][i], x[0][i]], axis=0) for i in range(len(x[0]))],
                tf.concat([s[1], x[1]], axis=0),
                tf.concat([s[2], x[2]], axis=0)]
        # need to increment indices on nlist
        snew[0][1] = tf.concat([s[0][1], x[0][1] + n], axis=0)
        n += tf.shape(x[0][0])[0]
        # need to make types match
        s = (tuple(snew[0]), *snew[1:])

    # state, (if full, current)
    return (s, i + 1, n), (i % L == L - 1, s)


def reshape_mask(*x):
    return (x[0], x[1], x[2][None])


def load_data(tfrecords, validation, embeddings, sample=True, batch_size=None, ensemble=0):
    # load data and split into train/validation

    # need to load each tf record individually and split
    # so that we have equal representation in validation data
    data = None
    validation_data = None
    print(f'Loading from {len(tfrecords)} files')
    for tfr in tfrecords:
        d = nmrdata.dataset(tfr, embeddings=embeddings,
                            label_info=True)
        # get size and split
        ds = len(list(d))
        vs = int(validation * ds)
        print(
            f'Loaded {tfr} and found {ds} records. Will keep {vs} for validation')
        v = d.take(vs)
        d = d.skip(vs)
        if data is None:
            if sample:
                data = [d]
                validation_data = [v]
            else:
                data = d
                validation_data = v
        else:
            if sample:
                data += [d]
                validation_data += [v]
            else:
                data = data.concatenate(d)
                validation_data = validation_data.concatenate(v)
    if sample:
        if ensemble > 0:
            if batch_size is not None and batch_size != ensemble:
                raise ValueError(
                    f'batch_size ({batch_size}) must be equal to ensemble ({ensemble})')
            # just sample from datasets
            train_data = tf.data.Dataset.sample_from_datasets(
                data, seed=0).map(kdens.map_reshape(ensemble, is_batched=True))
            for t in train_data:
                break
            v = validation_data[0]
            for vi in validation_data[1:]:
                v.concatenate(vi)
            validation_data = v
        else:
            if batch_size is None:
                batch_size = len(tfrecords)
            # we're going to squash elements together to batch via concat

            def squash_fxn(s, x):
                return squash_batch(s, x, batch_size)

            # need to get first record for scan
            for x0 in d:
                break

            train_data = tf.data.Dataset.sample_from_datasets(
                data, seed=0).scan((x0, 0, 0), squash_fxn)
            validation_data = tf.data.Dataset.sample_from_datasets(
                validation_data, seed=0).scan((x0, 0, 0), squash_fxn)

            # now we need to only keep the full element
            # we use indicator inserted during scan and then remove it
            train_data = train_data.filter(
                lambda b, x: b).map(lambda b, x: x)
            validation_data = validation_data.filter(
                lambda b, x: b).map(lambda b, x: x)
    else:
        # shuffle train at each iteration
        train_data = data.shuffle(500, reshuffle_each_iteration=True)
        if ensemble > 0:
            raise NotImplementedError()
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
