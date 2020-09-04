import click
import nmrdata
import tensorflow as tf
from .model import GNNModel, GNNHypers


@click.group()
def main():
    pass


@main.command()
@click.argument('tfrecords')
@click.argument('epochs', default=3)
@click.option('--embeddings', default=None, help='path to embeddings')
@click.option('--validation', default=0.2, help='relative size of validation')
def train(tfrecords, epochs, embeddings, validation):
    '''Train the model'''

    # load peak standards
    standards = nmrdata.load_standards()

    # load data and split into train/validation
    data = nmrdata.dataset(tfrecords, embeddings=embeddings).prefetch(
        tf.data.experimental.AUTOTUNE)
    data_size = len(list(data))
    validation_size = int(validation * data_size)
    validation_data, train_data = data.take(
        validation_size), data.skip(validation_size)
    hypers = GNNHypers()
    model = GNNModel(hypers, standards)

    # set-up learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=4, min_lr=1e-6)

    # compile with MSLE (to treat vastly different label mags)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_logarithmic_error',
                  metrics=[tf.keras.metrics.MeanAbsoluteError()]
                  )
    model.fit(train_data, epochs=epochs, callbacks=[
              reduce_lr], validation_data=validation_data)


if __name__ == '__main__':
    train()
