import time

import numpy as np
import pandas as pd
import tensorflow as tf

from autooc.algorithm.mapper import mapper
from autooc.algorithm.parameters import params


class Individual(object):
    """
    A GE individual.
    """

    def __init__(self, genome, ind_tree, map_ind=True):
        """
        Initialise an instance of the individual class (i.e. create a new
        individual).

        :param genome: An individual's genome.
        :param ind_tree: An individual's derivation tree, i.e. an instance
        of the representation.tree.Tree class.
        :param map_ind: A boolean flag that indicates whether or not an
        individual needs to be mapped.
        """

        if map_ind:
            # The individual needs to be mapped from the given input
            # parameters.
            (
                self.phenotype,
                self.genome,
                self.tree,
                self.nodes,
                self.invalid,
                self.depth,
                self.used_codons,
            ) = mapper(genome, ind_tree)

        else:
            # The individual does not need to be mapped.
            self.genome, self.tree = genome, ind_tree

        self.fitness = params["FITNESS_FUNCTION"].default_fitness
        self.runtime_error = False
        self.name = None,
        self.id = f"ID_{int(time.time()*1000000)}"
        self.training_time = 0
        self.predict_time = 0
        self.num_params = 0
        self.model = None
        self.threshold = 0
        self.losses = None
        self.rec_error = 0

    def __lt__(self, other):
        """
        Set the definition for comparison of two instances of the individual
        class by their fitness values. Allows for sorting/ordering of a
        population of individuals. Note that numpy NaN is used for invalid
        individuals and is used by some fitness functions as a default fitness.
        We implement a custom catch for these NaN values.

        :param other: Another instance of the individual class (i.e. another
        individual) with which to compare.
        :return: Whether or not the fitness of the current individual is
        greater than the comparison individual.
        """

        if np.isnan(self.fitness):
            return True
        elif np.isnan(other.fitness):
            return False
        else:
            return (
                self.fitness < other.fitness
                if params["FITNESS_FUNCTION"].maximise
                else other.fitness < self.fitness
            )

    def __le__(self, other):
        """
        Set the definition for comparison of two instances of the individual
        class by their fitness values. Allows for sorting/ordering of a
        population of individuals. Note that numpy NaN is used for invalid
        individuals and is used by some fitness functions as a default fitness.
        We implement a custom catch for these NaN values.

        :param other: Another instance of the individual class (i.e. another
        individual) with which to compare.
        :return: Whether or not the fitness of the current individual is
        greater than or equal to the comparison individual.
        """

        if np.isnan(self.fitness):
            return True
        elif np.isnan(other.fitness):
            return False
        else:
            return (
                self.fitness <= other.fitness
                if params["FITNESS_FUNCTION"].maximise
                else other.fitness <= self.fitness
            )

    def __str__(self):
        """
        Generates a string by which individuals can be identified. Useful
        for printing information about individuals.

        :return: A string describing the individual.
        """
        return "Individual: " + str(self.phenotype) + "; " + str(self.fitness)

    def deep_copy(self):
        """
        Copy an individual and return a unique version of that individual.

        :return: A unique copy of the individual.
        """

        if not params["GENOME_OPERATIONS"]:
            # Create a new unique copy of the tree.
            new_tree = self.tree.__copy__()

        else:
            new_tree = None

        # Create a copy of self by initialising a new individual.
        new_ind = Individual(self.genome.copy(), new_tree, map_ind=False)

        # Set new individual parameters (no need to map genome to new
        # individual).
        new_ind.phenotype, new_ind.invalid = self.phenotype, self.invalid
        new_ind.depth, new_ind.nodes = self.depth, self.nodes
        new_ind.used_codons = self.used_codons
        new_ind.runtime_error = self.runtime_error

        return new_ind

    def evaluate(self):
        """
        Evaluates phenotype in using the fitness function set in the params
        dictionary. For regression/classification problems, allows for
        evaluation on either training or test distributions. Sets fitness
        value.

        :return: Nothing unless multi-core evaluation is being used. In that
        case, returns self.
        """

        # Evaluate fitness using specified fitness function.
        self.fitness = params["FITNESS_FUNCTION"](self)

        if params["MULTICORE"]:
            return self

    def predict(self, X, anomaly_class, normal_class, threshold):
        #print(self.model.__class__.__name__)
        if self.model.__class__.__name__ == "Sequential":
            preds = self.predict_autoencoder(
                X, anomaly_class, normal_class, threshold)
        elif self.model.__class__.__name__ in ["OneClassSVM", "IsolationForest", "LocalOutlierFactor"]:
            preds = self.predict_sklearn(X, anomaly_class, normal_class)

        return preds

    def predict_autoencoder(self, X, anomaly_class, normal_class, threshold):
        reconstructions = self.reconstruct(X)
        loss = tf.keras.losses.mae(reconstructions, X)
        anomaly_mask = pd.Series(loss) > threshold
        preds = anomaly_mask.map(
            lambda x: anomaly_class if x == True else normal_class)
        #preds = tf.math.less(loss, self.threshold).numpy() * 1

        return preds

    def predict_sklearn(self, X, anomaly_class, normal_class):
        """_summary_

        Args:
            X (_type_): _description_
            anomaly_class (int, str): _description_
            normal_class (int, str): _description_

        Returns:
            _type_: _description_
        """

        preds = self.model.predict(X)
        preds = [(lambda x: anomaly_class if x == -1 else normal_class)(x)
                 for x in preds]

        return preds

    def reconstruct(self, X):
        """ Use the individual autoencoder to reconstruct the data.

        Args:
            X (np.array): data to reconstruct

        Returns:
            np.array: reconstructed dataset
        """
        reconstructions = self.model.predict(X)

        return reconstructions

    def get_id(self):
        return self.id
