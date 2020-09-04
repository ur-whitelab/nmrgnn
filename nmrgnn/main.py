import click
import nmrdata
import tensorflow as tf
from .model import GNNModel, GNNHypers


@click.command()
@click.argument('tfrecords')
@click.option('--embeddings', default=None, help='path to embeddings')
def train(tfrecords, embeddings):
    '''Train the model'''
    standards = nmrdata.load_standards()
    data = nmrdata.dataset(tfrecords, embeddings=embeddings).prefetch(
        tf.data.experimental.AUTOTUNE)
    hypers = GNNHypers()
    model = GNNModel(hypers)

    model.compile(optimizers='adam', loss='MeanSquaredError')


if __name__ == '__main__':
    train()
