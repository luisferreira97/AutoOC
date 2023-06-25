from autooc.algorithm.parameters import params
from autooc.fitness.base_ff_classes.base_ff import base_ff


class training_time(base_ff):
    """
    Fitness function class for minimising the training time of the autoencoder
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        return ind.training_time
