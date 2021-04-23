'''
Chemical shift predictor with graph neural network
'''
from .version import __version__
from .version import __release__

# by Luca Cappelletti (MIT)
def silence_tensorflow():    
    '''Silence every warning of notice from tensorflow.'''
    import logging, os
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ['KMP_AFFINITY'] = 'noverbose'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)

if __release__:
    silence_tensorflow()

from .layers import *
from .model import *
from .metrics import *
from .losses import *

custom_things = [NameRMSD, NameCorr, MPLayer, NameLoss, NameCount,
                 RBFExpansion, EdgeFCBlock, MPBlock, FCBlock]
custom_objects = {o.__name__: o for o in custom_things}
del custom_things

from .library import load_model, universe2graph, check_peaks
