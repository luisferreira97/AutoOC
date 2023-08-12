import json
import time
from os import path
from xml.dom.pulldom import parseString

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, Lambda
from tensorflow.keras.utils import plot_model

from autooc.algorithm.parameters import params
from autooc.fitness.base_ff_classes.base_ff import base_ff
from autooc.stats.stats import stats
from autooc.utilities.fitness.error_metric import (auc_autoencoder, auc_sklearn,
                                                  reconstruction_error)
from autooc.utilities.fitness.get_data import get_data
from autooc.utilities.fitness.optimize_constants import optimize_constants
from autooc.utilities.utils import (get_model_from_encoder,
                                   get_model_from_encoder2)

np.seterr(all="raise")


class supervised_learning(base_ff):
    """
    Fitness function for supervised learning, ie regression and
    classification problems. Given a set of training or test data,
    returns the error between y (true labels) and yhat (estimated
    labels).

    We can pass in the error metric and the dataset via the params
    dictionary. Of error metrics, eg RMSE is suitable for regression,
    while F1-score, hinge-loss and others are suitable for
    classification.

    This is an abstract class which exists just to be subclassed:
    should not be instantiated.
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Get training and test data
        # self.training_in, self.training_exp, self.test_in, self.test_exp = \
        # self.dtrain, self.dval = params['normal_train_data'], params['test_data']
        self.dtrain, self.dval = params["X_train"], params["X_val"]
        # get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

    def evaluate(self, ind, **kwargs):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :param kwargs: An optional parameter for problems with training/test
        data. Specifies the distribution (i.e. training or test) upon which
        evaluation is to be performed.
        :return: The fitness of the evaluated individual.
        """

        if params["ALGORITHM"] in ["autoencoder", "vae", "nas"]:
            metric = evaluate_autoencoder(ind, self.dtrain)
        elif params["ALGORITHM"] in ["iforest", "svm", "lof"]:
            metric = evaluate_sklearn(ind, self.dtrain)
        elif params["ALGORITHM"] in "all":
            metric = evaluate_all(ind, self.dtrain)

        return metric


def evaluate_autoencoder(ind, X_train):
    shape = X_train.shape[1]

    d = {
        "Sequential": Sequential,
        "Dense": Dense,
        "Lambda": Lambda,
        "BatchNormalization": BatchNormalization,
        "Input": Input,
        "Dropout": Dropout,
        "get_model_from_encoder": get_model_from_encoder2,
        "input_shape": shape,
    }

    exec(ind.phenotype, d)
    print("FENOTIPO")
    print(ind.phenotype)
    print("END FENOTIPO")
    #print(type(ind.phenotype))
    #print("\nSTART D:\n")
    #print(d)
    #print("\nEND D:\n")

    model = d["model"]

    # print("MODELO:")
    # model = get_model_from_encoder(ind.phenotype)
    # print(model)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    fit_start = time.time()
    hist = model.fit(
        X_train,
        X_train,
        epochs=params["EPOCHS"],
        # validation_data=(self.dval, self.dval),
        validation_split=0.25,
        verbose=1,
        callbacks=[early_stop],
    )
    fit_end = time.time()

    training_time = fit_end - fit_start
    ind.training_time = training_time

    predict_start = time.time()
    single_pred = model.predict(X_train[0:1])
    predict_end = time.time()
    predict_time = predict_end - predict_start

    ind.predict_time = predict_time
    ind.num_params = model.count_params()
    ind.model = model

    # save model history
    rand_name = str(stats["gen"]) + "_" + str(ind.name) + "_hist.json"
    filename = path.join(params["FILE_PATH"], rand_name)
    with open(filename, "w") as f:
        json.dump(hist.history, f)

    # save model plot
    print(model.summary())
    rand_name = str(stats["gen"]) + "_" + str(ind.name) + "_model.png"
    filename = path.join(params["FILE_PATH"], rand_name)
    plot_model(
        model, filename, show_shapes=True, show_dtype=True, expand_nested=True
    )

    if params["TYPE"] == "unsupervised":
        metric, rec_losses, threshold = metric_rec_error(model, ind)
    else:
        metric, rec_error, rec_losses, threshold = metric_auc(model, ind)

    ind.threshold = threshold
    ind.losses = rec_losses
    ind.rec_error = metric

    # save complexity scores
    filename = path.join(params["FILE_PATH"], "complexity.csv")
    with open(filename, "a") as the_file:
        the_file.write(
            (
                str(stats["gen"])
                + ";"
                + str(ind.name)
                + ";"
                + str(training_time)
                + "\n"
            )
        )

    #print(metric)
    return metric


def evaluate_sklearn(ind, X_train, algorithm=None):
    if params["ALGORITHM"] == "iforest" or algorithm == "iforest":
        d = {"IsolationForest": IsolationForest}
    elif params["ALGORITHM"] == "svm" or algorithm == "svm":
        d = {"OneClassSVM": OneClassSVM}
    elif params["ALGORITHM"] == "lof" or algorithm == "lof":
        d = {"LocalOutlierFactor": LocalOutlierFactor}

    exec(ind.phenotype, d)
    print("FENOTIPO")
    print(ind.phenotype)
    print("END FENOTIPO")
    #print(type(ind.phenotype))
    #print("\nSTART D:\n")
    #print(d)
    #print("\nEND D:\n")

    model = d["model"]

    fit_start = time.time()
    model.fit(X_train)
    fit_end = time.time()

    if params["TYPE"] == "unsupervised":
        metric = metric_anomaly_score(model, ind)
    else:
        metric = metric_auc_sklearn(model, ind)

    training_time = fit_end - fit_start
    ind.training_time = training_time

    predict_start = time.time()
    single_pred = model.predict(X_train[0:1])
    predict_end = time.time()
    predict_time = predict_end - predict_start

    ind.predict_time = predict_time
    #ind.num_params = model.count_params()
    ind.model = model
    ind.generation = stats["gen"]

    # save complexity scores
    filename = path.join(params["FILE_PATH"], "complexity.csv")
    with open(filename, "a") as the_file:
        the_file.write(
            (
                str(stats["gen"])
                + ";"
                + str(ind.name)
                + ";"
                + str(ind.id)
                + ";"
                + str(training_time)
                + "\n"
            )
        )

    #print(metric)
    return metric


def evaluate_all(ind, X_train):
    shape = X_train.shape[1]
    d = {
        "Sequential": Sequential,
        "Dense": Dense,
        "Lambda": Lambda,
        "BatchNormalization": BatchNormalization,
        "Input": Input,
        "Dropout": Dropout,
        "get_model_from_encoder": get_model_from_encoder2,
        "input_shape": shape,
        "OneClassSVM": OneClassSVM,
        "IsolationForest": IsolationForest,
        "LocalOutlierFactor": LocalOutlierFactor
    }

    exec(ind.phenotype, d)
    print("FENOTIPO")
    print(ind.phenotype)
    print("END FENOTIPO")
    # print(type(ind.phenotype))
    #print("\nSTART D:\n")
    #print(d)
    #print("\nEND D:\n")

    model = d["model"]
    #print(dir(model))
    # print(type(model.__class__.__name__))

    if model.__class__.__name__ == "Sequential":
        metric = evaluate_autoencoder(ind, X_train)
    elif model.__class__.__name__ == "OneClassSVM":
        metric = evaluate_sklearn(ind, X_train, algorithm="svm")
    elif model.__class__.__name__ == "IsolationForest":
        metric = evaluate_sklearn(ind, X_train, algorithm="iforest")
    elif model.__class__.__name__ == "LocalOutlierFactor":
        metric = evaluate_sklearn(ind, X_train, algorithm="lof")

    return metric


def metric_auc(model, ind):
    # calculate reconstruction error
    #auc = params["ERROR_METRIC"](model)
    auc, rec_error, rec_losses, threshold = auc_autoencoder(model)

    # save prediction scores
    filename = path.join(params["FILE_PATH"], "val_aucs.csv")
    with open(filename, "a") as the_file:
        the_file.write(
            (str(stats["gen"]) + ";" + str(ind.name) +
                ";" + str(ind.id) + ";" + str(auc) + "\n")
        )

    return auc, rec_error, rec_losses, threshold


def metric_rec_error(model, ind):
    # calculate reconstruction error
    rec_error, rec_losses, threshold = params["ERROR_METRIC"](model)

    # save prediction scores
    filename = path.join(params["FILE_PATH"], "val_rec_errors.csv")
    with open(filename, "a") as the_file:
        the_file.write(
            (str(stats["gen"]) + ";" + str(ind.name) +
                ";" + str(ind.id) +
                ";" + str(rec_error) + "\n")
        )

    return rec_error, rec_losses, threshold


def metric_auc_sklearn(model, ind):
    # calculate reconstruction error
    # auc = params["ERROR_METRIC"](model)
    auc = auc_sklearn(model)

    # save prediction scores
    filename = path.join(params["FILE_PATH"], "val_aucs.csv")
    with open(filename, "a") as the_file:
        the_file.write(
            (str(stats["gen"]) + ";" + str(ind.name) +
                ";" + str(ind.id) +
                ";" + str(auc) + "\n")
        )

    return auc


def metric_anomaly_score(model, ind):
    # calculate anomaly score
    #auc = params["ERROR_METRIC"](model)
    auc = params["ERROR_METRIC"](model)

    # save prediction scores
    filename = path.join(params["FILE_PATH"], "val_anomaly_scores.csv")
    with open(filename, "a") as the_file:
        the_file.write(
            (str(stats["gen"]) + ";" + str(ind.name) +
                ";" + str(ind.id) + ";" + str(auc) + "\n")
        )

    return auc
