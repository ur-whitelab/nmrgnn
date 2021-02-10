import click
import nmrdata
import tensorflow as tf
import kerastuner as kt
import nmrgnn
import pandas as pd
import numpy as np


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



def load_data(tfrecords, validation, embeddings, scale=False):
    # load data and split into train/validation
    data = nmrdata.dataset(tfrecords, embeddings=embeddings, label_info=True).prefetch(
        tf.data.experimental.AUTOTUNE)
    data_size = len(list(data))
    validation_size = int(validation * data_size)
    peak_standards = nmrdata.load_standards()
    peak_std = np.ones(100, dtype=np.float32)
    peak_avg = np.zeros(100, dtype=np.float32)
    for k, v in peak_standards.items():
        peak_std[k] = v[2]
        peak_avg[k] = v[1]
    validation_data = data.take(validation_size).cache()
    if scale:
        train_data = data.skip(validation_size).map(
            lambda *x: unstandardize_labels(*x, peak_std=peak_std, peak_avg=peak_avg)
        ).cache()
    else:
        train_data = data.skip(validation_size).cache()

    # shuffle train at each iteration
    train_data = train_data.shuffle(500, reshuffle_each_iteration=True)
    return train_data, validation_data


def setup_optimizations():
    #tf.debugging.enable_check_numerics()

    tf.config.optimizer.set_jit(True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_policy(policy)




@main.command()
@click.argument('tfrecords', nargs=-1, type=click.Path(exists=True))
@click.argument('epochs', default=3)
@click.option('--checkpoint-path', default='/tmp/checkpoint', type=click.Path(), help='where to save model')
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.2, help='relative size of validation')
@click.option('--tensorboard', default=None, help='path to tensorboard logs')
@click.option('--load/--noload', default=False, help='Load saved model at checkpoint path?')
@click.option('--loss-balance', default=1.0, help='Balance between L2 (max @ 1.0) and corr loss (max @ 0.0)')
def train(tfrecords, epochs, embeddings, validation, checkpoint_path, tensorboard, load, loss_balance):
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
    model.fit(train_data, epochs=epochs, callbacks=callbacks,
              validation_data=validation_data)



@main.command()
@click.argument('tfrecords')
@click.argument('checkpoint')
@click.argument('output')
@click.option('--validation', default=0.0, help='relative size of validation. If non-zero, only validation will be saved')
def eval_tfrecords(tfrecords, checkpoint, validation, output):
    '''Evaluate specific file'''    
    
    setup_optimizations()

    model = nmrgnn.build_GNNModel(metrics=False)
    model.load_weights(checkpoint)
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

    print(model.count_params())
    print(model.summary())


    out = pd.DataFrame({
        'element': element,
        'y': shift,
        'yhat': prediction,
        'class': class_name,
        'name': name
    })
    out.to_csv(f'{output}.csv', index=False)

@main.command()
@click.argument('tfrecords')
@click.argument('output_csv')
@click.argument('checkpoint')
def eval_pdb(pdbfile, output, checkpoint):
    '''Evaluate specific file'''    
    
    setup_optimizations()

    model = nmrgnn.build_GNNModel(metrics=False)
    model.load_weights(checkpoint)
    train_data, validation_data = load_data(tfrecords, 0.0, None)
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

    print(model.count_params())
    print(model.summary())


    out = pd.DataFrame({
        'element': element,
        'y': shift,
        'yhat': prediction,
        'class': class_name,
        'name': name
    })
    out.to_csv(f'{output}.csv', index=False)


@main.command()
@click.argument('tfrecords')
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
