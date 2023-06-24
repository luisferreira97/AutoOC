from talos.talos import Talos

# define problem
talos = Talos(anomaly_class=0,
              normal_class=1,
              multiobjective=True,
              performance_metric="training_time",
              algorithm="autoencoder"
              )

# load data
X_train, X_val, X_test, y_test = talos.load_example_data()

# train
run = talos.fit(
    X=X_train,
    X_val=X_val,
    pop=3,
    gen=3,
    experiment_name="software-impacts",
    epochs=100
)

# predict
predictions = talos.predict(X_test,
                            mode="all",
                            threshold="default")

# score
score = talos.evaluate(X_test,
                       y_test,
                       mode="all",
                       metric="roc_auc",
                       threshold="default")

print(score)
