import math
import os
import pickle
from datetime import datetime
from re import A
from typing import List

import mlflow
import numpy as np
import pandas as pd
#import pygmo as pg
from mlflow.tracking import MlflowClient
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from autooc.algorithm.parameters import params, set_params
from autooc.fitness.evaluation import evaluate_fitness
from autooc.operators.initialisation import initialisation
from autooc.stats.stats import get_stats, stats
from autooc.utilities.algorithm.general import check_python_version
from autooc.utilities.algorithm.initialise_run import initialise_run_params
from autooc.utilities.algorithm.NSGA2 import compute_pareto_metrics

check_python_version()


class AutoOC(object):
    def __init__(self, anomaly_class: str, normal_class: str, multiobjective: bool = False, algorithm: str = "all", performance_metric: str = "num_params", multicore: bool = False):
        #self.params = params.copy()
        self.params = params
        self.population = []
        self.population_dict = {}
        self.stats = {}
        self.fitted = False
        self.anomaly_class = anomaly_class
        self.normal_class = normal_class
        self.multiobjective = multiobjective
        self.algorithm = algorithm
        self.performance_metric = performance_metric
        self.multicore = multicore
        self.leaderboard = None

        init_params = {
            "ANOMALY_CLASS": self.anomaly_class,
            "NORMAL_CLASS": self.normal_class,
            "ALGORITHM": self.algorithm,
            "MULTICORE": self.multicore,
            "FITNESS_FUNCTION": f"supervised_learning.classification, {self.performance_metric}" if self.multiobjective else "supervised_learning.classification"
        }

        params.update(init_params)
        self.params.update(init_params)

    def fit(
        self,
        X: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame = None,
        pop: int = 100,
        gen: int = 100,
        early_stopping_rounds: int = None,
        early_stopping_tolerance: float = 0.01,
        always_at_hand: bool = False,
        results_path: str = "./",
        mlflow_tracking_uri: str = "./",
        mlflow_experiment_name: str = "mlflow_experiment",
        mlflow_run_name: str = f"run",
        **extra_params
    ) -> None:

        ALGORITHMS = {
            "autoencoder": {
                "grammar": os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammars", "autoencoders_v4.pybnf"),
                "metric_supervised": "auc_autoencoder",
                "metric_unsupervised": "reconstruction_error"
            },
            "iforest": {
                "grammar": os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammars", "iforest.pybnf"),
                "metric_supervised": "auc_sklearn",
                "metric_unsupervised": "anomaly_score_iforest"
            },
            "svm": {
                "grammar": os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammars", "svm.pybnf"),
                "metric_supervised": "auc_sklearn",
                "metric_unsupervised": "anomaly_score_svm"
            },
            "lof": {
                "grammar": os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammars", "lof.pybnf"),
                "metric_supervised": "auc_sklearn",
                "metric_unsupervised": "anomaly_score_lof"
            },
            "vae": {
                "grammar": os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammars", "vae.pybnf"),
                "metric_supervised": "auc_autoencoder",
                "metric_unsupervised": "reconstruction_error"
            },
            "nas": {
                "grammar": os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammars", "nas.pybnf"),
                "metric_supervised": "auc_autoencoder",
                "metric_unsupervised": "reconstruction_error"
            },
            "all": {
                "grammar": os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammars", "multi_algo.pybnf"),
                "metric_supervised": "auc_autoencoder",
                "metric_unsupervised": "reconstruction_error"
            }
        }

        fit_params = {
            "X_train": X,
            "X_train_len": len(X),
            "X_val": X_val,
            "y_val": y_val,
            "POPULATION_SIZE": pop,
            "GENERATIONS": gen,
            "EXPERIMENT_NAME": mlflow_experiment_name,
            "TYPE": "unsupervised" if y_val is None else "supervised",
            # "FITNESS_FUNCTION": f"supervised_learning.classification, {self.performance_metric}" if self.multiobjective else "supervised_learning.classification",
            "RESULTS_PATH": os.path.abspath(results_path),
            "ALWAYS_AT_HAND": always_at_hand,
        }

        if self.algorithm in ALGORITHMS:
            fit_params["GRAMMAR_FILE"] = ALGORITHMS[self.algorithm]["grammar"]
            fit_params["ERROR_METRIC"] = ALGORITHMS[
                self.algorithm][f"metric_{fit_params['TYPE']}"]
        else:
            fit_params["GRAMMAR_FILE"] = ALGORITHMS["all"]["grammar"]
            fit_params["ERROR_METRIC"] = ALGORITHMS["all"][f"metric_{fit_params['TYPE']}"]

        if self.multiobjective:
            fit_params["REPLACEMENT"] = "nsga2_replacement"
            fit_params["SELECTION"] = "nsga2_selection"
        else:
            fit_params["REPLACEMENT"] = "generational"
            fit_params["SELECTION"] = "tournament"

        params.update(fit_params)
        self.params.update(fit_params)

        param_list = list_params(**extra_params)
        set_params(param_list)

        #self.params = params

        # initialize run
        initialise_run_params(True)

        # initialise population
        self.population = initialisation(self.params["POPULATION_SIZE"])

        # Evaluate initial population
        self.population = evaluate_fitness(self.population)

        stats["gen"] = 0

        # Generate statistics for run so far
        get_stats(self.population)

        #mlflow = get_mlflow(self.params["EXPERIMENT_NAME"])
        mlflow_params = {
            "run_name": mlflow_run_name,
            "tracking_uri": mlflow_tracking_uri,
            "experiment_name": mlflow_experiment_name,
        }

        range_generations = tqdm(range(1, (self.params["GENERATIONS"] + 1)))

        self.population = self.evolve(
            self.params, range_generations, self.population, early_stopping_rounds, early_stopping_tolerance, mlflow_params)
        # get_stats(population, end=True)
        store_pop(self.population)

        #print(self.population)
        self.population_dict = {ind.id: ind for ind in self.population}

        self.stats = stats
        #self.population = population
        self.fitted = True

    def reconstruct(self, X_test: pd.DataFrame, mode: str = "all"):
        if mode == "all":
            reconstrutions = [ind.reconstruct(X_test)
                              for ind in self.population]
        elif mode == "best":
            if self.multiobjective:
                best = min(self.population, key=lambda x: x.fitness[0])
            else:
                best = min(self.population, key=lambda x: x.fitness)
            reconstrutions = best.reconstruct(X_test)
        elif mode == "simplest":
            simplest = min(self.population, key=lambda x: x.fitness[1])
            reconstrutions = simplest.reconstruct(X_test)
        elif mode == "balanced":
            from math import log10

            min_y = log10(
                min(self.population, key=lambda x: x.fitness[1]).fitness[1])
            max_y = log10(
                max(self.population, key=lambda x: x.fitness[1]).fitness[1])
            # get individual with greater distance to point (0, 1)
            balanced = max(self.population,
                           key=lambda x: get_distance(x, min_y, max_y))
            reconstrutions = balanced.reconstuct(X_test)

        return reconstrutions

    def predict(self, X_test: pd.DataFrame, mode: str = "all", threshold="default", percentile=95) -> list:
        individuals = self.get_individuals(mode)
        if threshold == "default":
            return [ind.predict(X_test, self.anomaly_class, self.normal_class, ind.threshold) for ind in individuals]

        elif type(threshold) in [int, float]:
            return [ind.predict(X_test, self.anomaly_class, self.normal_class, threshold) for ind in individuals]

        else:
            thresholds = self.calculate_threshold(
                mode, threshold, percentile=percentile)
            return [individuals[i].predict(X_test, self.anomaly_class, self.normal_class, thresholds[i]) for i in range(len(individuals))]

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        mode: str = "all",
        metric: str = "roc_auc",
        threshold="default",
        percentile=95
    ) -> list:
        metrics = {
            "roc_auc": roc_auc_score,
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        }

        preds = self.predict(X_test, mode, threshold, percentile)

        if isinstance(self.anomaly_class, str):
            preds = [(lambda x: x.replace(self.anomaly_class, '1'))(x)
                     for x in preds]
            preds = [(lambda x: x.replace(self.normal_class, '0'))(x)
                     for x in preds]
            #print(preds[0])
            preds = [(lambda x: x.astype(int))(x) for x in preds]

            y_test = (lambda x: x.replace(self.anomaly_class, '1'))(y_test)
            y_test = (lambda x: x.replace(self.normal_class, '0'))(y_test)
            #print(y_test)
            y_test = (lambda x: x.astype(int))(y_test)

        scores = [metrics[metric](y_test, pred) for pred in preds]

        return scores

    def evaluate_all(self, X_test: pd.DataFrame, y_test: pd.Series, metric: str = "roc_auc") -> List:
        """
        Evaluate all individuals in terms of AUC and Complexity,\
            based on data set (X_test, y_test)
        Only works for multiobjetive problems

        Parameters
        ----------
        X_test : pd.DataFrame
            The test input samples.
        y_test : pd.Series
            The test target values (class labels) as integers or strings.

        Returns
        -------
        List
            List containing [AUC, Complexity] for each Evolutionary Decision Tree in the population.
        """
        performances = self.evaluate(X_test, y_test, metric=metric, mode="all")
        # performances = [metrics[metric](y_test, ind.predict(X_test, self.anomaly_class, self.normal_class))
        #        for ind in self.population]

        complexities = [ind.fitness[1] for ind in self.population]
        ev = pd.DataFrame([performances, complexities]).T
        ev.columns = ['performance', 'complexity']
        ev2 = ev.groupby('performance').agg(
            {'complexity': min}).reset_index().values

        return [list(ev2[:, 0]), list(ev2[:, 1])]

    def load_example_data(self) -> np.array:
        # Download the dataset
        PATH_TO_DATA = (
            "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
        )
        data = pd.read_csv(PATH_TO_DATA, header=None).add_prefix("col_")

        # last column is the target
        # 0 = anomaly, 1 = normal
        TARGET = "col_140"

        features = data.drop(TARGET, axis=1)
        target = data[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, stratify=target
        )

        train_index = y_train[y_train == 1].index
        train_data = X_train.loc[train_index]

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = min_max_scaler.fit_transform(train_data.copy())
        X_train_scaled, X_val_scaled = train_test_split(
            X_train_scaled, test_size=0.25)
        X_test_scaled = min_max_scaler.transform(X_test.copy())

        return X_train_scaled, X_val_scaled, X_test_scaled, y_test

    def split_data(self, df: pd.DataFrame, target_col: str, normalize: bool = True) -> pd.DataFrame:
        features = df.drop(target_col, axis=1)
        target = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, stratify=target
        )

        train_index = y_train[y_train == self.normal_class].index
        train_data = X_train.loc[train_index]

        if normalize:
            min_max_scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_scaled = min_max_scaler.fit_transform(train_data.copy())
            X_train_scaled, X_val_scaled = train_test_split(
                X_train_scaled, test_size=0.25)
            X_test_scaled = min_max_scaler.transform(X_test.copy())
            return X_train_scaled, X_val_scaled, X_test_scaled, y_test
        else:
            X_train, X_val = train_test_split(
                X_train, test_size=0.25)
            return X_train, X_val, X_test, y_test

    def calculate_threshold(self, mode: str = "all", method: str = "mean", percentile=95):
        individuals = self.get_individuals(mode)
        if method == "mean":
            thresholds = [(np.mean(ind.losses) + np.std(ind.losses))
                          for ind in individuals]
        elif method == "percentile":
            thresholds = [np.percentile(ind.losses, percentile)
                          for ind in individuals]
        elif method == "max":
            thresholds = [np.max(ind.losses) for ind in individuals]

        return thresholds

    def get_individuals(self, mode: str = "all") -> list:
        if mode == "all":
            individuals = self.population
        elif mode == "best":
            if self.params["ERROR_METRIC"].maximise:
                individuals = [max(self.population, key=lambda x: x.fitness[0])] if self.multiobjective else [
                    max(self.population, key=lambda x: x.fitness)]
            else:
                individuals = [min(self.population, key=lambda x: x.fitness[0])] if self.multiobjective else [
                    min(self.population, key=lambda x: x.fitness)]
        elif mode == "pareto":
            individuals = self.get_pareto_front()
        elif mode == "simplest":
            individuals = [min(self.population, key=lambda x: x.fitness[1])]
        return individuals

    def get_pareto_front(self) -> list:
        return compute_pareto_metrics(self.population).fronts[0]

    def get_statistics(self) -> dict:
        return self.stats

    def get_performance_metric(self) -> str:
        return self.performance_metric

    def get_population(self) -> dict:
        return self.population

    def get_population_dict(self) -> dict:
        return self.population_dict

    def get_params(self) -> dict:
        return self.params

    def get_classes(self) -> dict:
        return {
            "anomaly_class": self.anomaly_class,
            "normal_class": self.normal_class
        }

    def get_default_thresholds(self) -> list:
        return [ind.threshold for ind in self.population]

    def get_individual(self, id):
        return self.population_dict[id]

    def get_leaderboard(self, orderby="generation") -> dict:
        if orderby == "generation":
            return self.leaderboard
        elif orderby == "performance":
            return self.leaderboard.sort_values(by="reconstruction_error", ascending=True)
        elif orderby == "complexity":
            return self.leaderboard.sort_values(by=self.performance_metric, ascending=True)

    def set_classes(self, anomaly_class, normal_class) -> None:
        self.anomaly_class = anomaly_class
        self.normal_class = normal_class

    def evolve(
        self,
        params: dict,
        range_generations: range,
        population: list,
        early_stopping_rounds: int,
        early_stopping_tolerance: float,
        mlflow_params: dict = None,
    ) -> List:

        leaderboard_list = []

        # config mlflow experiment
        mlflow.set_tracking_uri(os.path.join(
            os.path.abspath(mlflow_params["tracking_uri"]), "mlruns"))
        mlflow.set_registry_uri(os.path.join(
            os.path.abspath(mlflow_params["tracking_uri"]), "mlruns"))
        mlflow.set_experiment(mlflow_params["experiment_name"])

        # trigger to end current run faster
        end = False
        round_count = 0  # rounds without improvement

        if self.params["ERROR_METRIC"].maximise:
            best_performance = 0
        else:
            best_performance = np.inf

        with mlflow.start_run(run_name=mlflow_params["run_name"]) as run:
            for key, val in params.items():
                if key in [
                    "X_train",
                    "y_train",
                    "X_train_len",
                    "X_val",
                    "y_val",
                    "X_test",
                    "y_test",
                    "POOL",
                    "BNF_GRAMMAR",
                    "SEED_INDIVIDUALS",
                ]:
                    continue
                else:
                    mlflow.log_param(key, val)

            for generation in range_generations:
                best_performance_before_generation = best_performance
                stats["gen"] = generation

                population = params["STEP"](population)

                i = 1
                for ind in population:
                    leaderboard_dict = {
                        "ID": ind.id,
                        "generation": generation,
                        "individual": i
                    }

                    if self.multiobjective:
                        leaderboard_dict[self.params["ERROR_METRIC"].__name__] = ind.fitness[0],
                        leaderboard_dict[self.performance_metric] = ind.fitness[1]
                    else:
                        leaderboard_dict[self.params["ERROR_METRIC"].__name__] = ind.fitness
                        #print("INDIVIDUAL FITNESS: \n\n\n\n\n\n\n", ind.fitness)
                        if self.params["ERROR_METRIC"].maximise:
                            if ind.fitness > best_performance:
                                best_performance = ind.fitness
                        else:
                            if ind.fitness < best_performance:
                                best_performance = ind.fitness

                    leaderboard_list.append(leaderboard_dict)
                    i += 1

                if self.multiobjective:
                    performances = [ind.fitness[0] for ind in population]
                    complexities = [ind.fitness[1] for ind in population]
                    generation_metrics = {
                        "1st ind PERFORMANCE": min(performances),
                        "1st ind COMPLEXITY": max(complexities),
                        "last ind PERFORMANCE": max(performances),
                        "last ind COMPLEXITY": min(complexities),
                        "mean PERFORMANCE": np.mean(performances),
                        "mean COMPLEXITY": np.mean(complexities),
                    }
                else:
                    performances = [ind.fitness for ind in population]
                    generation_metrics = {
                        "1st ind PERFORMANCE": min(performances),
                        "last ind PERFORMANCE": max(performances),
                        "mean PERFORMANCE": np.mean(performances)
                    }

                mlflow.log_metrics(
                    metrics=generation_metrics,
                    step=generation,
                )

                if self.multiobjective:
                    # get pareto front from the current generation
                    pareto_individuals = compute_pareto_metrics(
                        population).fronts[0]
                    if self.params["ERROR_METRIC"].maximise:
                        pareto_points = [[-ind.fitness[0], ind.fitness[1]]
                                         for ind in pareto_individuals]
                    else:
                        pareto_points = [[ind.fitness[0], ind.fitness[1]]
                                         for ind in pareto_individuals]

                    #hypervolume = pg.hypervolume(pareto_points)
                    #ref_value = hypervolume.refpoint()
                    #hypervolume_value = hypervolume.compute(ref_value)
                    #print("HYPERVOLUME: \n\n\n\n\n\n\n\n\n\n\n", hypervolume_value)

                    # if hypervolume_value > best_performance:
                    #    best_performance = hypervolume_value

                    if (best_performance + early_stopping_tolerance) > best_performance_before_generation:
                        round_count = 0
                    else:
                        round_count += 1
                else:
                    if self.params["ERROR_METRIC"].maximise:
                        if (best_performance + early_stopping_tolerance) > best_performance_before_generation:
                            round_count = 0
                        else:
                            round_count += 1
                    else:
                        if (best_performance - early_stopping_tolerance) < best_performance_before_generation:
                            round_count = 0
                            best_performance_before_generation = best_performance
                        else:
                            round_count += 1

                #print(f"ROUND COUNT \n\n\n\n\n\n\:{round_count}")

                if early_stopping_rounds is not None and round_count >= early_stopping_rounds:
                    end = True
                    #print("EARLY STOPPING\n\n\n\n\n\n\n")

                if generation == params["GENERATIONS"] or end:
                    i = 0
                    for ind in population:
                        if ind.model.__class__.__name__ == "Sequential":
                            #mlflow.keras.log_model(ind.model, f"model_{i}")
                            pass
                        elif ind.model.__class__.__name__ in ["OneClassSVM", "IsolationForest"]:
                            #mlflow.sklearn.log_model(ind.model, f"model_{i}")
                            pass
                        i += 1

                    if end:
                        break

            get_stats(population, end=True)
            artifacts_dir = params["FILE_PATH"]
            mlflow.log_artifacts(artifacts_dir, artifact_path="images")

            # save leaderboard
            leaderboard = pd.DataFrame.from_dict(leaderboard_list)
            leaderboard.to_csv(params["FILE_PATH"] +
                               "/leaderboard.csv", index=False)

            mlflow.log_artifact(params["FILE_PATH"] + "/leaderboard.csv")
            self.leaderboard = leaderboard.copy()

        # mlflow.end_run()

        return population


def list_params(**extra_params: dict) -> List:
    """
    Internal function to list execution parameters.
    For advanced users only.
    Parameters
    ----------
    **extra_params : dict
        Extra parameters. For details, please check: https://github.com/PonyGE/PonyGE2/wiki/Evolutionary-Parameters.
    Returns
    -------
    param_list : List
        List of parameters.
    """
    param_list = []
    for key, val in extra_params.items():
        if val == "True":
            param_list.append("--" + key)
        elif val in ["False", ""]:
            continue
        else:
            param_list.append("--{0}={1}".format(key, str(val)))
    return param_list


def store_pop(population: List):
    """
    Stores the evolved population. For advanced users only.
    Parameters
    ----------
    population : List
        The population to be stored
    Returns
    -------
    None.
    """

    # SEEDS_PATH = path.join('results', params["FILE_PATH"], 'seeds')
    # makedirs(path.join(getcwd(), SEEDS_PATH, params['TARGET_SEED_FOLDER']),
    #         exist_ok=True)
    SEEDS_PATH = os.path.join(params["FILE_PATH"], "seeds")
    os.makedirs(SEEDS_PATH, exist_ok=True)
    for cont, item in enumerate(population):
        if item.phenotype != None:
            # fname = path.join(SEEDS_PATH, params['TARGET_SEED_FOLDER'],
            #                  "{0}.txt".format(str(cont)))
            fname = os.path.join(
                SEEDS_PATH, params["TARGET_SEED_FOLDER"], "{0}.txt".format(
                    str(cont))
            )
            with open(fname, "w+", encoding="utf-8") as f:
                f.write("Phenotype:\n")
                f.write("%s\n" % item.phenotype)
                f.write("Genotype:\n")
                f.write("%s\n" % item.genome)
                f.write("Tree:\n")
                f.write("%s\n" % str(item.tree))
                f.write("Training fitness:\n")
                f.write("%s\n" % item.fitness)
                f.close()


def get_distance(ind, min_y, max_y):
    auc = ind.fitness[0] / -100  # auc (positive, from 0 to 1)
    comp = math.log10(ind.fitness[1])  # complexity
    # scale complexity to [0, 1]
    comp = (comp - min_y) / (max_y - min_y)
    # worst result: (0, 1)
    x = 0
    y = 1
    # get distance:
    dist = math.hypot(auc - x, comp - y)
    return dist
