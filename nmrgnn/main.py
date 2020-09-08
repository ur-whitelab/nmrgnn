import click
import nmrdata
import tensorflow as tf
import kerastuner as kt
from .model import GNNModel, GNNHypers


@click.group()
def main():
    pass


@main.command()
@click.argument('tfrecords')
@click.argument('epochs', default=3)
@click.option('--checkpoint_path', default='/tmp/checkpoint', help='where to save model')
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.2, help='relative size of validation')
@click.option('--tensorboard', default=None, help='path to tensorboard logs')
def train(tfrecords, epochs, embeddings, validation, checkpoint_path, tensorboard):
    '''Train the model'''

    # load peak standards
    standards = nmrdata.load_standards()

    # load data and split into train/validation
    data = nmrdata.dataset(tfrecords, embeddings=embeddings).prefetch(
        tf.data.experimental.AUTOTUNE)
    data_size = len(list(data))
    validation_size = int(validation * data_size)
    validation_data, train_data = data.take(
        validation_size).cache(), data.skip(validation_size).cache()

    # shuffle train
    train_data = train_data.shuffle(500, reshuffle_each_iteration=True)

    hypers = GNNHypers()

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
    
        model = GNNModel(hypers, standards)

        # compile with MSLE (to treat vastly different label mags)
        optimizer = tf.keras.optimizers.Adam(1e-4)
        model.compile(optimizer=optimizer,
                  loss='mean_squared_logarithmic_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError()]
                  )

    callbacks = []
    # set-up learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=4, min_lr=1e-6)
    callbacks.append(reduce_lr)
    # tensorboard
    if tensorboard is not None:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard, write_images=False, write_graph=False, histogram_freq=0, profile_batch=0)
        callbacks.append(tensorboard_callback)
    # save model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor='val_loss',
        save_best_only=True)
    callbacks.append(model_checkpoint_callback)

    tuner = RandomSearch(
        model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=3,
        directory='nmrgnn',
        project_name='nmrgnn')
    #model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
    tuner.search(train_data,
        epochs=epochs,
        validation_data=validation_data)

    best_model = tuner.get_best_models()[0]
    best_model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=validation_data)

if __name__ == '__main__':
    train()
