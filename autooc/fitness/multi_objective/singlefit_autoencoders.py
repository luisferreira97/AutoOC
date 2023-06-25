#!/usr/bin/env python
"""
@author: Jonathan Byrne
@17/01/18 11:09
"""

import json
import time
from os import path

import numpy as np
# imports relative to keras model (ind.phenotype)
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.utils import plot_model

from autooc.algorithm.parameters import params
from autooc.fitness.base_ff_classes.base_ff import base_ff
from autooc.stats.stats import stats
from autooc.utilities.utils import get_model_from_encoder


class singlefit_autoencoders(base_ff):
    """
    An example of a single fitness class that generates
    two fitness values for multi-objective optimisation
    """

    maximise = True
    multi_objective = True

    def __init__(self):

        # Initialise base fitness function class.
        super().__init__()

        # Set list of individual fitness functions.
        self.num_obj = 2
        dummyfit = base_ff()
        dummyfit.maximise = True
        self.fitness_functions = [dummyfit, dummyfit]
        self.default_fitness = [float("nan"), float("nan")]

        self.dtrain, self.dval = params["normal_train_data"], params["test_data"]

    def evaluate(self, ind, **kwargs):
        """Dummy fitness function that generates 2 fitness values"""
        phenotype = ind.phenotype
        fitness = 0
        settings = {}

        d = {
            "Sequential": Sequential,
            "Dense": Dense,
            "BatchNormalization": BatchNormalization,
            "Input": Input,
            "Dropout": Dropout,
            "get_model_from_encoder": get_model_from_encoder,
        }

        exec(ind.phenotype, d)
        print("FENOTIPO")
        print(ind.phenotype)
        print("END FENOTIPO")
        # print(type(ind.phenotype))
        print("\nSTART D:\n")
        print(d)
        print("\nEND D:\n")

        model = d["model"]

        rand_name = str(stats["gen"]) + "_" + str(ind.name) + "_model.png"
        filename = path.join(params["FILE_PATH"], rand_name)
        plot_model(model, filename, show_shapes=True)
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )
        start = time.time()
        hist = model.fit(
            self.dtrain,
            self.dtrain,
            epochs=params["EPOCHS"],
            validation_data=(self.dval, self.dval),
            verbose=1,
            callbacks=[early_stop],
        )
        end = time.time()
        # params["TRAINING_TIME"] = end-start
        training_time = start - end
        rand_name = str(stats["gen"]) + "_" + str(ind.name) + "_hist.json"
        filename = path.join(params["FILE_PATH"], rand_name)
        with open(filename, "w") as f:
            json.dump(hist.history, f)

        auc = params["ERROR_METRIC"](model)

        filename = path.join(params["FILE_PATH"], "test_aucs.csv")
        with open(filename, "a") as the_file:
            the_file.write(
                (str(stats["gen"]) + ";" +
                 str(ind.name) + ";" + str(auc) + "\n")
            )
        print(auc)

        fitness = [auc, training_time]

        return fitness

    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.

        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vector.
        """

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]
