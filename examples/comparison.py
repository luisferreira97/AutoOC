from autooc.autooc import AutoOC

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

"""
AutoOC execution with supervised mode
"""

# load data
train = pd.read_csv("data/credit-card-supervised-train.csv")
val = pd.read_csv("data/credit-card-supervised-val.csv")
test = pd.read_csv("data/credit-card-supervised-test.csv")

target = "Class"

X_train = train.drop(columns=target).values
y_train = train[target].values

X_val = val.drop(columns=target).values
y_val = val[target].values

y_test = test[target].values
X_test = test.drop(columns=target).values

# define problem
aoc = AutoOC(anomaly_class=0,
    normal_class=1,
    multiobjective=True,
    performance_metric="training_time",
    algorithm="all"
)

# train (using 20 individuals and 100 generations, similar to our previous comparison study)
run = aoc.fit(
    X=X_train,
    X_val=X_val,
    y_val=y_val,
    pop=20,
    gen=100,
    epochs=100,
    mlflow_tracking_uri="../results",
    mlflow_experiment_name="comparison_supervised",
    mlflow_run_name="comparison_supervised_run",
    results_path="../results"
)

# predict
predictions = aoc.predict(X_test,
                            mode="all",
                            threshold="default")

# score
score = aoc.evaluate(X_test,
                       y_test,
                       mode="all",
                       metric="roc_auc",
                       threshold="default")

print(score)

"""
AutoOC execution with unsupervised mode
"""

# load data
train = pd.read_csv("data/credit-card-unsupervised-train.csv")
val = pd.read_csv("data/credit-card-unsupervised-val.csv")
test = pd.read_csv("data/credit-card-unsupervised-test.csv")

target = "Class"

X_train = train.drop(columns=target).values
y_train = train[target].values

X_val = val.drop(columns=target).values
y_val = val[target].values

y_test = test[target].values
X_test = test.drop(columns=target).values

# define problem
aoc = AutoOC(anomaly_class=0,
    normal_class=1,
    multiobjective=True,
    performance_metric="training_time",
    algorithm="all"
)

# train (using 20 individuals and 100 generations, similar to our previous comparison study)
run = aoc.fit(
    X=X_train,
    X_val=X_val,
    pop=20,
    gen=100,
    epochs=100,
    mlflow_tracking_uri="../results",
    mlflow_experiment_name="comparison_unsupervised",
    mlflow_run_name="comparison_unsupervised_run",
    results_path="../results"
)

# predict
predictions = aoc.predict(X_test,
    mode="all",
    threshold="default"
)

# score
score = aoc.evaluate(X_test,
    y_test,
    mode="all",
    metric="roc_auc",
    threshold="default"
)

print(score)

"""
Isolation Forest execution
"""

from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=500)
clf.fit(X_train)
y_pred = clf.predict(X_test)

# transform 1 into 0 and -1 into 1 (to be equal to AutoOC)
y_pred = [(lambda x: 0 if x == 1 else 1)(x) for x in y_pred]

# get auc
print(roc_auc_score(y_test, y_pred))
