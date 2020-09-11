import click
import nmrdata
import tensorflow as tf
import kerastuner as kt
from .model import *


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
        if load:
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            model = build_GNNModel()
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
        save_weights_only=False,
        monitor='val_loss',
        save_best_only=False)
    callbacks.append(model_checkpoint_callback)

    train_data, validation_data = load_data(tfrecords, validation, embeddings)
    model.fit(train_data, epochs=epochs, callbacks=callbacks,
              validation_data=validation_data)
    model.save(checkpoint_path)


@main.command()
@click.argument('tfrecords')
@click.argument('epochs', default=3)
@click.option('--tuning_path', default='tuning', help='where to save tuning information')
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.2, help='relative size of validation')
@click.option('--tensorboard', default=None, help='path to tensorboard logs')
def hyper(tfrecords, epochs, embeddings, tuning_path, validation, tensorboard):
    '''Tune hyperparameters the model'''

    callbacks = []
    # set-up learning rate scheduler
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
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

    train_data, validation_data = load_data(tfrecords, validation, embeddings)

    tuner = kt.tuners.hyperband.Hyperband(
        build_GNNModel,
        objective=kt.Objective('val_loss', direction='min'),
        max_epochs=epochs,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        executions_per_trial=3,
        directory=tuning_path,
        project_name='gnn-tuning')

    tuner.search(train_data,
                 validation_data=validation_data,
                 callbacks=callbacks)
    tuner.search_space_summary()


if __name__ == '__main__':
    main()
