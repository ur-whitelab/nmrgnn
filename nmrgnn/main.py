import click
import nmrdata
import tensorflow as tf
import kerastuner as kt
import nmrgnn
import pandas as pd


@click.group()
def main():
    pass


def load_data(tfrecords, validation, embeddings):
    # load data and split into train/validation
    data = nmrdata.dataset(tfrecords, embeddings=embeddings, label_info=True).prefetch(
        tf.data.experimental.AUTOTUNE)
    data_size = len(list(data))
    validation_size = int(validation * data_size)
    validation_data, train_data = data.take(
        validation_size).cache(), data.skip(validation_size).cache()
    # shuffle train
    train_data = train_data.shuffle(500, reshuffle_each_iteration=True)
    return train_data, validation_data


@main.command()
@click.argument('tfrecords')
@click.argument('epochs', default=3)
@click.option('--checkpoint-path', default='/tmp/checkpoint', help='where to save model')
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.2, help='relative size of validation')
@click.option('--tensorboard', default=None, help='path to tensorboard logs')
@click.option('--load/--noload', default=False, help='Load saved model at checkpoint path?')
def train(tfrecords, epochs, embeddings, validation, checkpoint_path, tensorboard, load):
    '''Train the model'''

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = nmrgnn.build_GNNModel()
        if load:
            model.load_weights(checkpoint_path)
    callbacks = []
    # set-up learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=4, min_lr=1e-6, verbose=1)
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

    train_data, validation_data = load_data(tfrecords, validation, embeddings)
    model.fit(train_data, epochs=epochs, callbacks=callbacks,
              validation_data=validation_data)

@main.command()
@click.argument('tfrecords')
@click.argument('epochs', default=3)
@click.option('--checkpoint-path', default='/tmp/checkpoint', help='where to save model')
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.2, help='relative size of validation')
@click.option('--tensorboard', default=None, help='path to tensorboard logs')
@click.option('--load/--noload', default=False, help='Load saved model at checkpoint path?')
def train(tfrecords, epochs, embeddings, validation, checkpoint_path, tensorboard, load):
    '''Train the model'''

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = nmrgnn.build_GNNModel()
        if load:
            model.load_weights(checkpoint_path)
    callbacks = []
    # set-up learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=4, min_lr=1e-6, verbose=1)
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

    train_data, validation_data = load_data(tfrecords, validation, embeddings)
    model.fit(train_data, epochs=epochs, callbacks=callbacks,
              validation_data=validation_data)



@main.command()
@click.argument('tfrecords')
@click.argument('checkpoint')
@click.argument('output')
@click.option('--validation', default=0.0, help='relative size of validation. If non-zero, only validation will be saved')
def eval_tfrecords(tfrecords, checkpoint, validation, output):
    '''Tune hyperparameters the model'''    
    
    model = nmrgnn.build_GNNModel()
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
        break
        print(f'\rComputing...{count}')
    out = pd.DataFrame({
        'element': element,
        'y': shift,
        'yhat': prediction,
        'class': class_name,
        'name': name
    })
    out.to_csv(f'{output}.csv', index=False)
    
if __name__ == '__main__':
    main()
