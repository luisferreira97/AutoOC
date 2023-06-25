from autooc.algorithm.fitness.error_metric import calculate_bic
from autooc.algorithm.parameters import params
from autooc.fitness.base_ff_classes.base_ff import base_ff


class bic(base_ff):
    """
    Fitness function class for calculating Bayesian Information Criterion
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        return calculate_bic(ind.model, params["X_train_len"])
