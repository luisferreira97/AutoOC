import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from talos.talos import Talos

if __name__ == "__main__":

    #talos = Talos(anomaly_class = 0, normal_class = 1, multiobjective=True, performance_metric="num_params", multicore=False)
talos = Talos(anomaly_class=0, normal_class=1, multiobjective=True,
              performance_metric="training_time", multicore=False, algorithm="autoencoder")
talos = Talos(anomaly_class=0, normal_class=1, multiobjective=True,
              performance_metric="training_time", multicore=False, algorithm="iforest")
talos = Talos(anomaly_class=0, normal_class=1, multiobjective=True,
              performance_metric="training_time", multicore=False, algorithm="svm")
talos = Talos(anomaly_class=0, normal_class=1,
              multiobjective=False, algorithm="svm")


talos = Talos(anomaly_class=0, normal_class=1, multiobjective=True,
              performance_metric="training_time", multicore=False, algorithm="iforest")
X_train, X_val, X_test, y_test = talos.load_example_data()

run = talos.fit(
    X=X_train,
    X_val=X_test,
    y_val=y_test,
    #X_val = X_val,
    #y_val = y_val,
    pop=5,
    gen=20,
    # multiobjective=True,
    # performance_metric="num_params",
    # multicore=False,
    experiment_name="file_path_2",
    results_path="talos",
    early_stopping=2,
    epochs=1000
)

"""talos = Talos(anomaly_class="Anomaly", normal_class="Normal",
                  multiobjective=True, performance_metric="predict_time", multicore=False)
    df = pd.read_csv("/home/lferreira/talos/data/speech.csv")
    X_train, X_val, X_test, y_test = talos.split_data(
        df, target_col="Target", normalize=False)"""

"""talos = Talos(anomaly_class=1, normal_class=0, multiobjective=True,
                    performance_metric="num_params", multicore=False)
    df = pd.read_csv("/home/lferreira/talos/data/credit-card-fraud.csv")
    df = df.iloc[:20000, :]
    X_train, X_val, X_test, y_test = talos.split_data(
        df, target_col="Class")"""

talos = Talos(anomaly_class=1, normal_class=0, multiobjective=True,
              performance_metric="num_params", multicore=False)
df = pd.read_csv(
    "/home/lferreira/CMMS-ASOC-2023/data/failure_prediction_final.csv")
targets = ["DaysToNextFailure", "FailOn3Days",
           "FailOn5Days", "FailOn7Days", "FailOn10Days"]
target = "FailOn3Days"  # <- selecionar a target a considerar na experiência!
targets.remove(target)
df.drop(columns=targets, inplace=True)

df.fillna(0, inplace=True)

train, test = train_test_split(df, stratify=df[target])


train_normal = train[train[target] == 0]
train_anomaly = train[train[target] == 1]

train_normal = train.drop(index=train_anomaly.index)

train, val = train_test_split(train_normal)

X_train = train.drop(columns=target).values
y_train = train[target].values

X_val = val.drop(columns=target).values
y_val = val[target].values

test = test.append(train_anomaly)
X_test = test.drop(columns=target).values
y_test = test[target].values

"""normal_sample = train_normal.sample(n=50)
train_normal = train_normal.drop(index=normal_sample.index)

val = train[train[target] == 1]
val = X_val.append(normal_sample)

X_val = val.drop(columns=target).values
y_val = val[target].values


anomaly_data = """

talos = Talos(anomaly_class=1, normal_class=0, multiobjective=True,
              performance_metric="num_params", multicore=False)
talos = Talos(anomaly_class=1, normal_class=0, multiobjective=True,
              performance_metric="training_time", multicore=False, algorithm="svm")
target = "FailOn3Days"  # <- selecionar a target a considerar na experiência!
train = pd.read_csv(
    "/home/lferreira/CMMS-ASOC-2023/src/transmogrifai-class-3/fold1/train.csv")
train.fillna(0, inplace=True)
test = pd.read_csv(
    "/home/lferreira/CMMS-ASOC-2023/src/transmogrifai-class-3/fold1/test.csv")
test.fillna(0, inplace=True)

# cols_to_keep = ['RecordType_FAILURE', 'WOType_CORRETIVA',
#       'WOType_PREVENTIVA', 'PriorityLevel_MUITO URGENTE',
#       'PriorityLevel_URGENTE', 'TotalTime', 'Quantity',
#        'AssetWithFailure', 'RecordDay',
#       'DaysAfterPurchaseDate', 'FailOn3Days']

#train = train[cols_to_keep]
#test = test[cols_to_keep]

y_test = test[target].values
X_test = test.drop(columns=target).values

val = train[train[target] == 1]
train = train[train[target] == 0]
normal_sample = train.sample(n=len(val))
train = train.drop(index=normal_sample.index)

X_train = train.drop(columns=target).values
y_train = train[target].values

val = val.append(normal_sample)
X_val = val.drop(columns=target).values
y_val = val[target].values

print(X_train.shape)

run = talos.fit(
    X=X_train,
    #X_val = X_test,
    #y_val = y_test,
    X_val=X_val,
    y_val=y_val,
    pop=5,
    gen=5,
    # multiobjective=True,
    # performance_metric="num_params",
    # multicore=False,
    experiment_name="ocsvm",
    epochs=1000
)
print(run)

preds = talos.predict(X_test, mode="all", threshold="default")

#roc_auc_score(y_test, preds[0])
print(len(preds))

score = talos.evaluate(X_test, y_test, mode="all", metric="roc_auc")
print(score)

scores = talos.evaluate_all(X_test, y_test, metric="roc_auc")
print(scores)
