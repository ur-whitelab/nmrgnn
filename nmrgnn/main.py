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
    return x, tf.stack([w * new_labels, y[:, 1], y[:,-1]], axis=1), w


def _load_baseline():
    from importlib_resources import files
    import nmrgnn.models
    fp = files(nmrgnn.models).joinpath(
        'baseline')
    return fp


def check_peaks(atoms, peaks, cutoff_sigma=4, warn_sigma=2.5):
    peak_standards = nmrdata.load_standards()    
    confident = np.empty(atoms.shape[0], dtype=np.bool)
    confident[:] = True
    for i in range(len(atoms)):
        ps = peak_standards[int(np.nonzero(atoms[i])[0])]        
        if ps[2] == 0 or (peaks[i] - ps[1])**2 / ps[2]**2 > warn_sigma**2:
            confident[i] = False
        #if ps[2] == 0 or (peaks[i] - ps[1])**2 / ps[2]**2 > cutoff_sigma**2:
        #    peaks[i] = np.nan
    return peaks, confident
            
        


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
        print(f'Loaded {tfr} and found {ds} records. Will keep {vs} for validation')
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
            lambda *x: unstandardize_labels(*x, peak_std=peak_std, peak_avg=peak_avg)
        )

    # shuffle train at each iteration
    train_data = data.shuffle(500, reshuffle_each_iteration=True)
    return train_data.prefetch(tf.data.experimental.AUTOTUNE), validation_data.cache()


def setup_optimizations():
    #tf.debugging.enable_check_numerics()

    tf.config.optimizer.set_jit(True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_policy(policy)




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
    
    train_data, validation_data = load_data(tfrecords, validation, embeddings, scale=False)

    # explicitly call model to get shapes defined 
    for t in train_data:
        x,y,m = t
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
        model_file = _load_baseline()
    model_name = os.path.basename(model_file)

    model = tf.keras.models.load_model(model_file, custom_objects=nmrgnn.custom_objects)
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
    rev_names = {v: k for k,v in embeddings['name'].items()}
    for x,y,w in data:
        # get predictions
        yhat = model(x)
        ytrue = y[:, 0]
        namei = y[:,1]#tf.cast(y[:,1], tf.int32)
        name.extend([rev_names[int(n)].split('-')[1] for wi,n in zip(w, namei) if wi > 0])
        class_name.extend([rev_names[int(n)].split('-')[0] for wi,n in zip(w, namei) if wi > 0])
        element.extend([rev_names[int(n)].split('-')[1][0] for wi,n in zip(w, namei) if wi > 0])
        prediction.extend([float(yi) for wi,yi in zip(w, yhat) if wi > 0])
        shift.extend([float(yi) for wi,yi in zip(w, ytrue) if wi > 0])
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
        results[f'{data_name}-{e}-r'].append(out[out.element == e].corr().iloc[0,1])
    for n in np.unique(out.name):
        results[f'{data_name}-{n}-r'] = [len(out[out.name == n].y)]
        results[f'{data_name}-{n}-r'].append(out[out.name == n].corr().iloc[0,1])
    for e in np.unique(out.element):
        results[f'{data_name}-{e}-rmsd'] = [len(out[out.element == e].y)]
        results[f'{data_name}-{e}-rmsd'].append(np.mean((out[out.element == e].yhat - out[out.element == e].y)**2))
    for n in np.unique(out.name):
        results[f'{data_name}-{n}-rmsd'] = [len(out[out.name == n].y)]
        results[f'{data_name}-{n}-rmsd'].append(np.mean((out[out.name == n].yhat - out[out.name == n].y)**2))
    results = pd.DataFrame(results, index=['N', model_name])
    results = results.transpose()

    if merge is None:
        merge = f'{model_name}.md'
    else:
        # https://stackoverflow.com/a/60156036
        # read markdopwn table
        if os.path.exists(merge):
            other = pd.read_table(merge, sep="|", header=0, index_col=1, skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
            # remove whitespace in column names
            other.columns = other.columns.str.replace(' ','')
            results = pd.concat([results, other])
        
    with open(merge, 'w') as f:
        f.write(results.to_markdown())
        f.write('\n')

@main.command()
@click.argument('struct-file')
@click.argument('output-csv')
@click.option('--model-file', type=click.Path(exists=True), default=None, help='Model file. If not provided, baseline will be used.')
@click.option('--neighbor-number', default=16, help='The model specific size of neighbor lists')
def eval_struct(struct_file, output_csv, model_file, neighbor_number):
    '''Predict NMR chemical shifts with specific file'''    

    import nmrdata.parse
    
    setup_optimizations()

    if model_file is None:
        model_file = _load_baseline()
    model_name = os.path.basename(model_file)

    model = tf.keras.models.load_model(model_file, custom_objects=nmrgnn.custom_objects)

    embeddings = nmrdata.load_embeddings()
    
    import MDAnalysis as md
    u = md.Universe(struct_file)

    atoms, edges, nlist = nmrdata.parse.parse_universe(u, neighbor_number, embeddings) 
    mask = np.ones_like(atoms)
    inv_degree = tf.squeeze(tf.math.divide_no_nan(1.,
                                                  tf.reduce_sum(tf.cast(nlist > 0, tf.float32), axis=1)))

    peaks = model((atoms, nlist, edges, inv_degree))
    peaks, confident = check_peaks(atoms.numpy(), peaks.numpy())
    
    out = pd.DataFrame({
        'index': np.arange(atoms.shape[0]),
        'names': u.atoms.names,
        'peaks': np.round(peaks, 2),
        'confident': confident
    })
    out.to_csv(f'{output_csv}', index=False)


@main.command()
@click.argument('tfrecords', nargs=-1, type=click.Path(exists=True))
@click.argument('epochs', default=3)
@click.option('--tuning_path', default='tuning', help='where to save tuning information')
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.1, help='relative size of validation')
@click.option('--tensorboard', default=None, help='path to tensorboard logs')
def hyper(tfrecords, epochs, embeddings, tuning_path, validation, tensorboard):
    '''Tune hyperparameters the model'''

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

    train_data, validation_data = load_data(tfrecords, validation, embeddings, scale=False)

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

    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    model = tuner.hypermodel.build(best_hps)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=4, min_lr=1e-6, verbose=1)
    callbacks = [reduce_lr]

    model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=validation_data)    


if __name__ == '__main__':
    main()
