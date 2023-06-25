import warnings
from math import log

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm

from autooc.algorithm.parameters import params


def mae(y, yhat):
    """
    Calculate mean absolute error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """

    return np.mean(np.abs(y - yhat))


# Set maximise attribute for mae error metric.
mae.maximise = False


def rmse(y, yhat):
    """
    Calculate root mean square error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The root mean square error.
    """

    return np.sqrt(np.mean(np.square(y - yhat)))


# Set maximise attribute for rmse error metric.
rmse.maximise = False


def mse(y, yhat):
    """
    Calculate mean square error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean square error.
    """

    return np.mean(np.square(y - yhat))


# Set maximise attribute for mse error metric.
mse.maximise = False


def hinge(y, yhat):
    """
    Hinge loss is a suitable loss function for classification.  Here y is
    the true values (-1 and 1) and yhat is the "raw" output of the individual,
    ie a real value. The classifier will use sign(yhat) as its prediction.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The hinge loss.
    """

    # Deal with possibility of {-1, 1} or {0, 1} class label convention
    y_vals = set(y)
    # convert from {0, 1} to {-1, 1}
    if 0 in y_vals:
        y[y == 0] = -1

    # Our definition of hinge loss cannot be used for multi-class
    assert len(y_vals) == 2

    # NB not np.max. maximum does element-wise max.  Also we use the
    # mean hinge loss rather than sum so that the result doesn't
    # depend on the size of the dataset.
    return np.mean(np.maximum(0, 1 - y * yhat))


# Set maximise attribute for hinge error metric.
hinge.maximise = False


def f1_score(y, yhat):
    """
    The F_1 score is a metric for classification which tries to balance
    precision and recall, ie both true positives and true negatives.
    For F_1 score higher is better.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The f1 score.
    """

    # if phen is a constant, eg 0.001 (doesn't refer to x), then yhat
    # will be a constant. that will break f1_score. so convert to a
    # constant array.
    if not isinstance(yhat, np.ndarray) or len(yhat.shape) < 1:
        yhat = np.ones_like(y) * yhat

    # Deal with possibility of {-1, 1} or {0, 1} class label
    # convention.  FIXME: would it be better to canonicalise the
    # convention elsewhere and/or create user parameter to control it?
    # See https://github.com/PonyGE/PonyGE2/issues/113.
    y_vals = set(y)
    # convert from {-1, 1} to {0, 1}
    if -1 in y_vals:
        y[y == -1] = 0

    # We binarize with a threshold, so this cannot be used for multi-class
    assert len(y_vals) == 2

    # convert real values to boolean {0, 1} with a zero threshold
    yhat = yhat > 0

    with warnings.catch_warnings():
        # if we predict the same value for all samples (trivial
        # individuals will do so as described above) then f-score is
        # undefined, and sklearn will give a runtime warning and
        # return 0. We can ignore that warning and happily return 0.
        warnings.simplefilter("ignore")
        return sklearn_f1_score(y, yhat, average="weighted")


# Set maximise attribute for f1_score error metric.
f1_score.maximise = True


def AUC(y, yhat):
    """
    Calculate the Area Under the Curve (AUC).
    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The AUC.
    """
    ### NEW ###
    # Replaces NaNs or infinite values
    yhat = np.nan_to_num(yhat)
    if type(yhat) != np.ndarray:
        yhat = np.repeat(yhat, len(y))
    # auc_val = roc_auc_score(y, yhat)*100
    auc_score = roc_auc_score(y, yhat)
    # auc_val = auc(fpr, tpr) * 100
    return auc_score

    # print(auc)
    # return(auc)


# Set maximise attribute for auc error metric.
AUC.maximise = True

"""def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))
    print("AUC = {}".format(roc_auc_score(labels, predictions)))

preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)"""


def auc_autoencoder(model):
    rec_error, rec_losses, threshold = reconstruction_error(model)

    X_val = params["X_val"]
    y_val = params["y_val"]

    #predictions = tf.math.less(rec_losses, threshold)
    predictions = [(lambda x: params["ANOMALY_CLASS"] if x >
                    threshold else params["NORMAL_CLASS"])(x) for x in rec_losses]

    try:
        auc = roc_auc_score(y_val, predictions)
    except Exception:
        auc = 0
    # print("AUCCCCCC")
    print(auc)

    return auc, rec_error, rec_losses, threshold


auc_autoencoder.maximise = True


def auc_sklearn(model):
    X_val = params["X_val"]
    y_val = params["y_val"]

    predictions = model.predict(X_val)
    predictions = [(lambda x: params["ANOMALY_CLASS"] if x == -
                    1 else params["NORMAL_CLASS"])(x) for x in predictions]

    print(predictions)
    print(y_val)

    try:
        auc = roc_auc_score(y_val, predictions)
    except Exception:
        auc = 0
    print("AUCCCCCC")
    print(auc)

    return auc


auc_sklearn.maximise = True


def reconstruction_error(model):
    test_data = params["X_val"]
    print(test_data.shape)

    reconstructions = model.predict(test_data)
    print(reconstructions.shape)
    rec_losses = tf.keras.losses.mae(reconstructions, test_data)

    threshold = np.mean(rec_losses) + np.std(rec_losses)

    rec_error = tf.math.reduce_mean(rec_losses).numpy()
    # threshold = 0.018963557
    # predictions = tf.math.less(test_loss, threshold)

    # test_labels = params["test_labels"]
    # auc = roc_auc_score(test_labels, predictions)

    return rec_error, rec_losses, threshold


reconstruction_error.maximise = False


def anomaly_score_sklearn(model):
    test_data = params["X_val"]
    print(test_data.shape)

    anomaly_scores = model.decision_function(test_data)
    print(anomaly_scores.shape)

    return np.mean(anomaly_scores)


anomaly_score_sklearn.maximise = False


def anomaly_score_iforest(model):
    test_data = params["X_val"]
    print(test_data.shape)

    anomaly_scores = model.decision_function(test_data)
    print(anomaly_scores.shape)

    return np.mean(anomaly_scores)


anomaly_score_iforest.maximise = False


def anomaly_score_svm(model):
    test_data = params["X_val"]
    print(test_data.shape)

    anomaly_scores = model.decision_function(test_data)
    print(anomaly_scores.shape)

    return np.mean(anomaly_scores)


anomaly_score_svm.maximise = True


def anomaly_score_lof(model):
    test_data = params["X_val"]
    print(test_data.shape)

    anomaly_scores = model.decision_function(test_data)
    print(anomaly_scores.shape)

    return np.mean(anomaly_scores)


anomaly_score_lof.maximise = True


# calculate Bayesian Information Criterion
def calculate_bic(model, n):
    rec_error, _ = reconstruction_error(model)
    return n * log(rec_error) + model.count_params() * log(n)


calculate_bic.maximise = False


def Hamming_error(y, yhat):
    """
    The number of mismatches between y and yhat. Suitable
    for Boolean problems and if-else classifier problems.
    Assumes both y and yhat are binary or integer-valued.
    """
    return np.sum(y != yhat)


Hamming_error.maximise = False
