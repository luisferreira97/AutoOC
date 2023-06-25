from autooc.algorithm.parameters import params
from autooc.fitness.base_ff_classes.base_ff import base_ff


class num_params(base_ff):
    """
    Fitness function class for minimising the number of parameters of the autoencoder
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        return ind.num_params
