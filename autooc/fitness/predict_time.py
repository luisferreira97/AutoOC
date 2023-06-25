from autooc.algorithm.parameters import params
from autooc.fitness.base_ff_classes.base_ff import base_ff


class predict_time(base_ff):
    """
    Fitness function class for minimising the predict time of the autoencoder
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        return ind.predict_time
