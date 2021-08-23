import click
import os
import logging
import pandas as pd
import numpy as np
import pickle


import tensorflow as tf
import kerastuner as kt
import nmrdata
import nmrgnn
from .library import *


@click.group()
def main():
    pass


@tf.function
def unstandardize_labels(x, y, w, peak_std, peak_avg):
    # expand to be size of one-hot elements
    num_elem = x[0].shape[1]
    peak_std = peak_std[:num_elem]
    peak_avg = peak_std[:num_elem]
    # shape N
    labels = y[:, 0]
    # shape N x Eelem
    nodes = x[0]
    new_labels = tf.math.divide_no_nan(
        labels - tf.reduce_sum(nodes * peak_avg, axis=1),
        tf.reduce_sum(nodes * peak_std, axis=1))
    return x, tf.stack([w * new_labels, y[:, 1], y[:, -1]], axis=1), w


@main.command()
@click.argument('tfrecords', nargs=-1, type=click.Path(exists=True))
@click.argument('name')
@click.argument('epochs', default=3)
@click.option('--checkpoint-path', default='/tmp/checkpoint', type=click.Path(), help='where to save model')
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.1, help='relative size of validation')
@click.option('--tensorboard', default=None, help='path to tensorboard logs')
@click.option('--load/--noload', default=False, help='Load saved model at checkpoint path?')
@click.option('--loss-balance', default=1.0, help='Balance between L2 (max @ 1.0) and corr loss (max @ 0.0)')
def train(tfrecords, name, epochs, embeddings, validation, checkpoint_path, tensorboard, load, loss_balance):
    '''Train the model'''

    model = nmrgnn.build_GNNModel(loss_balance=loss_balance)
    if load:
        model.load_weights(checkpoint_path)
    callbacks = []
    # set-up learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.99,
                                                     patience=4, min_lr=1e-4, verbose=1)
    callbacks.append(reduce_lr)
    # tensorboard
    if tensorboard is not None:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard, write_images=False, write_graph=False, histogram_freq=0, profile_batch=0)
        callbacks.append(tensorboard_callback)
    # save model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=False)
    callbacks.append(model_checkpoint_callback)

    train_data, validation_data = load_data(
        tfrecords, validation, embeddings, scale=False)

    # explicitly call model to get shapes defined
    for t in train_data:
        x, y, m = t
        model(x)
        break

    results = model.fit(train_data, epochs=epochs, callbacks=callbacks,
                        validation_data=validation_data, validation_freq=1)

    model.save(name)

    pfile = name + '-history-0.pb'
    i = 0
    while os.path.exists(pfile):
        i += 1
        pfile = f'{name}-history-{i}.pb'
    with open(pfile, 'wb') as f:
        pickle.dump(results.history, file=f)


@main.command()
@click.argument('tfrecords', nargs=-1, type=click.Path(exists=True))
@click.option('--model-file', type=click.Path(exists=True), default=None, help='Model file. If not provided, baseline will be used.')
@click.option('--validation', default=0.0, help='relative size of validation. If non-zero, only validation will be saved')
@click.option('--data-name', default='', help='Short name for data on table output')
@click.option('--merge', default=None, help='Merge results with another markdown table')
def eval_tfrecords(tfrecords, model_file, validation, data_name, merge):
    '''Evaluate specific file'''

    if model_file is None:
        model_file = load_baseline()
    model_name = os.path.basename(model_file)

    if tfrecords is None or len(tfrecords) == 0:
        raise ValueError('Must give input TFRecord files')

    model = tf.keras.models.load_model(
        model_file, custom_objects=nmrgnn.custom_objects)
    train_data, validation_data = load_data(tfrecords, validation, None)
    if validation > 0:
        data = validation_data
    else:
        data = train_data
    embeddings = nmrdata.load_embeddings()
    print('Computing...')
    element = []
    prediction = []
    shift = []
    name = []
    class_name = []
    count = 0
    rev_names = {v: k for k, v in embeddings['name'].items()}
    for x, y, w in data:
        # get predictions
        yhat = model(x)
        ytrue = y[:, 0]
        namei = y[:, 1]  # tf.cast(y[:,1], tf.int32)
        name.extend([rev_names[int(n)].split('-')[1]
                     for wi, n in zip(w, namei) if wi > 0])
        class_name.extend([rev_names[int(n)].split('-')[0]
                           for wi, n in zip(w, namei) if wi > 0])
        element.extend([rev_names[int(n)].split('-')[1][0]
                        for wi, n in zip(w, namei) if wi > 0])
        prediction.extend([float(yi) for wi, yi in zip(w, yhat) if wi > 0])
        shift.extend([float(yi) for wi, yi in zip(w, ytrue) if wi > 0])
        count += 1
        print(f'\rComputing...{count}', end='')
    print('done')

    # I think this just prints (no need for print?)
    model.summary()

    out = pd.DataFrame({
        'element': element,
        'y': shift,
        'yhat': prediction,
        'class': class_name,
        'name': name
    })
    out.to_csv(f'{model_name}.csv', index=False)

    # compute correlations & RMSD broken out by class
    results = dict()
    for e in np.unique(out.element):
        results[f'{data_name}-{e}-r'] = [len(out[out.element == e].y)]
        results[f'{data_name}-{e}-r'].append(
            out[out.element == e].corr().iloc[0, 1])
    for n in np.unique(out.name):
        results[f'{data_name}-{n}-r'] = [len(out[out.name == n].y)]
        results[f'{data_name}-{n}-r'].append(
            out[out.name == n].corr().iloc[0, 1])
    for e in np.unique(out.element):
        results[f'{data_name}-{e}-rmsd'] = [len(out[out.element == e].y)]
        results[f'{data_name}-{e}-rmsd'].append(
            np.mean((out[out.element == e].yhat - out[out.element == e].y)**2))
    for n in np.unique(out.name):
        results[f'{data_name}-{n}-rmsd'] = [len(out[out.name == n].y)]
        results[f'{data_name}-{n}-rmsd'].append(
            np.mean((out[out.name == n].yhat - out[out.name == n].y)**2))
    results = pd.DataFrame(results, index=['N', model_name])
    results = results.transpose()

    if merge is None:
        merge = f'{model_name}.md'
    else:
        # https://stackoverflow.com/a/60156036
        # read markdopwn table
        if os.path.exists(merge):
            other = pd.read_table(merge, sep="|", header=0, index_col=1,
                                  skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
            # remove whitespace in column names
            other.columns = other.columns.str.replace(' ', '')
            results = pd.concat([results, other])

    with open(merge, 'w') as f:
        f.write(results.to_markdown())
        f.write('\n')


@main.command()
@click.argument('struct-files', nargs=-1, type=click.Path(exists=True))
@click.argument('output-csv')
@click.option('--model-file', type=click.Path(exists=True), default=None, help='Model file. If not provided, baseline will be used.')
@click.option('--neighbor-number', default=16, help='The model specific size of neighbor lists')
@click.option('--stride', default=1, help='Stride for reading trajectory, if multiple frames are present')
def eval_struct(struct_files, output_csv, model_file, neighbor_number, stride):
    '''Predict NMR chemical shifts with specific file'''

    if len(struct_files) == 0:
        raise ValueError('Must pass at least on structure file')

    import nmrdata

    setup_optimizations()

    if model_file is None:
        model_file = load_baseline()
    model_name = os.path.basename(model_file)

    model = tf.keras.models.load_model(
        model_file, custom_objects=nmrgnn.custom_objects)

    embeddings = nmrdata.load_embeddings()

    import MDAnalysis as md
    import tqdm
    import time
    u = md.Universe(*struct_files)
    out = None
    N = len(u.trajectory)

    # add useful info
    gpus = tf.config.list_physical_devices('GPU')
    gpu_msg = 'GPUs: None'
    if len(gpus) > 0:
        gpu_msg = f'GPUs: {len(gpus)}'

    inf_msg = f'Model Inference ({gpu_msg})'
    timing = {'MDAnalysis': 0, inf_msg: 0, 'Parsing': 0}

    pbar = None
    if N > 1:
        pbar = tqdm.tqdm(total=N)
    for i, ts in enumerate(u.trajectory[::stride]):
        # i > 0 -> means only warn on 1st iter
        t = time.time_ns()
        atoms, edges, nlist = nmrdata.parse_universe(
            u, neighbor_number, embeddings, warn=i == 0)
        inv_degree = tf.squeeze(tf.math.divide_no_nan(1.,
                                                      tf.reduce_sum(tf.cast(nlist > 0, tf.float32), axis=1)))
        timing['MDAnalysis'] += time.time_ns() - t
        t = time.time_ns()
        peaks = model((atoms, nlist, edges, inv_degree))
        confident = check_peaks(atoms, peaks)

        timing[inf_msg] += time.time_ns() - t
        t = time.time_ns()
        data = pd.DataFrame({
            'index': np.arange(atoms.shape[0]),
            'residues': u.atoms.resnames,
            'resids': u.atoms.resids,
            'names': u.atoms.names,
            'peaks': np.round(peaks, 2),
            'confident': confident,
            'time': np.repeat(ts.time, atoms.shape[0]),
            'frame': np.repeat(ts.frame, atoms.shape[0])
        })

        if out is None:
            out = data
        else:
            out = pd.concat((out, data))

        timing['Parsing'] += time.time_ns() - t
        t = time.time_ns()
        if pbar:
            pbar.set_description(
                '|'.join([f'{k}:{v/10**9:5.2f}s' for k, v in timing.items()]))
            pbar.update(stride)
    out.to_csv(f'{output_csv}', index=False)
    print('|'.join([f'{k}:{v/10**9:5.2f}s' for k, v in timing.items()]))


@main.command()
@click.argument('tfrecords', nargs=-1, type=click.Path(exists=True))
@click.argument('epochs', default=3)
@click.option('--tuning_path', default='tuning', help='where to save tuning information')
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.1, help='relative size of validation')
@click.option('--tensorboard', default=None, help='path to tensorboard logs')
def hyper(tfrecords, epochs, embeddings, tuning_path, validation, tensorboard):
    '''Tune hyperparameters the model'''

    if tfrecords is None or len(tfrecords) == 0:
        raise ValueError('Must give input TFRecord files')

    setup_optimizations()

    callbacks = []
    # set-up learning rate scheduler
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0,
        mode='min',
        restore_best_weights=False,
    )

    callbacks.append(early_stop)
    # tensorboard
    if tensorboard is not None:
        print('Will write tensorboard summaries')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard, update_freq='epoch', write_images=False, write_graph=False, histogram_freq=0, profile_batch=0)
        callbacks.append(tensorboard_callback)

    train_data, validation_data = load_data(
        tfrecords, validation, embeddings, scale=False)

    tuner = kt.tuners.hyperband.Hyperband(
        nmrgnn.build_GNNModel,
        objective=kt.Objective('val_loss', direction='min'),
        max_epochs=epochs,
        hyperband_iterations=3,
        executions_per_trial=3,
        directory=tuning_path,
        project_name='gnn-tuning')

    tuner.search(train_data,
                 validation_data=validation_data,
                 callbacks=callbacks)
    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=4, min_lr=1e-6, verbose=1)
    callbacks = [reduce_lr]

    model.fit(train_data, epochs=epochs, callbacks=callbacks,
              validation_data=validation_data)


if __name__ == '__main__':
    main()
