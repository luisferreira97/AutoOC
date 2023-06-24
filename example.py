from talos.talos import Talos

talos = Talos(anomaly_class=0, normal_class=1, multiobjective=True,
              performance_metric="training_time", multicore=False, algorithm="iforest")
X_train, X_val, X_test, y_test = talos.load_example_data()

run = talos.fit(
    X=X_train,
    X_val=X_test,
    y_val=y_test,
    #X_val = X_val,
    #y_val = y_val,
    pop=10,
    gen=10,
    # multiobjective=True,
    # performance_metric="num_params",
    # multicore=False,
    experiment_name="early_stopping_test_multi",
    results_path="talos",
    #early_stopping = 3,
    #always_at_hand = False,
    epochs=1000
)
