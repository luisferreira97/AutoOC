<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!--[![Downloads](https://static.pepy.tech/personalized-badge/autooc?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/autooc)-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/luisferreira97/AutoOC">
    <img src="https://raw.githubusercontent.com/luisferreira97/AutoOC/main/images/logo.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">AutoOC (in Beta)</h3>

  <p align="center">
    AutoOC: Automated Machine Learning (AutoML) library focused on One-Class Learning algorithms (Deep AutoEncoders, Variational AutoEncoders, Isolation Forest, Local Outlier Factor and One-Class SVM)
    <br />
    <!--<a href="https://github.com/luisferreira97/AutoOC"><strong>Explore the docs »</strong></a>
    <br />
    <br />-->
    <!--<a href="https://github.com/luisferreira97/AutoOC">View Demo</a>
    ·-->
    <a href="https://github.com/luisferreira97/AutoOC/issues">Report Bug</a>
    ·
    <a href="https://github.com/luisferreira97/AutoOC/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS 
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>-->



<!-- ABOUT THE PROJECT 
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

There are many great README templates available on GitHub, however, I didn't find one that really suit my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should element DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have have contributed to expanding this template!

A list of commonly used resources that I find helpful are listed in the acknowledgements.
-->

<!-- GETTING STARTED -->
## Getting Started

This section presents how the package can be reached and installed.

### Where to get it

The source code is currently hosted on GitHub at: https://github.com/luisferreira97/AutoOC

Binary installer for the latest released version are available at the Python Package Index (PyPI). The PyPI name of the package is `autooc`.

```sh
pip install autooc
```

<!-- USAGE EXAMPLES -->
## Usage

### 1. Import the package
The first step in using the package is, after it has been installed, to import it. The main class from which all the methods are available is ```AutoOC```.

```python
from autooc.autooc import AutoOC
```

### 2. Instantiate a AutoOC object
The second step is to instantiate the AutoOC class with the information about your dataset and context (e.g., normal and anomaly classes, wether to run single-objective or multi-objective, the performance_metric, and the algorithm).
You can change the ```algorithm``` parameter to select which algorithms are used during the optimization. The options are:
- "autoencoders": Deep AutoEncoders (from TensorFlow)
- "vae": Variational AutoEncoders (from TensorFlow)
- "iforest": Isolation Forest (from Scikit-Learn)
- "lof": Local Outlier Factor (Scikit-Learn)
- "svm": One-Class SVM (from Scikit-Learn)
- "nas": the optimization is done using AutoEncoders and VAEs
- "all": the optimization is done using all five algorithms

For the ```performance_metric``` parameter to select which algorithms are used during the optimization. The options are:
- "training_time": Minimizes training time
- "predict_time": Minimizes the time it takes to predict one record
- "num_params": Minimizes the number of parameters (```count_params()``` in Keras); only available when ```algorithm``` equals to ```autoencoders```, ```vae```, or ```nas```.
- "bic": Minimizes the value of the Bayesian Information Criterion

```python
aoc = AutoOC(anomaly_class = 0,
    normal_class = 1,
    multiobjective=True,
    performance_metric="training_time",
    algorithm = "autoencoder"
)
```

### 3. Load dataset
The third step is to load the dataset. Depending on the type of validation you need *train data* (only 'normal' instances), *validation data* (you can use (1) only 'normal' instances or (2) both 'normal' and 'anomaly' instances with the respective labels), and *test data* (both types of instances and labels). You can use the ```load_example_data()``` function to load the popular ECG dataset.


```python
X_train, X_val, X_test, y_test = aoc.load_example_data()
```

### 4. Train
The fourth step is to train the model. The ```fit()``` function computes the optimization using the given parameters.

```python
run = aoc.fit(
    X=X_train,
    X_val=X_val,
    pop=3,
    gen=3,
    epochs=100,
    mlflow_tracking_uri="../results",
    mlflow_experiment_name="test_experiment",
    mlflow_run_name="test_run",
    results_path="../results"
)
```

### 5. Predict

The fifth step is to predict the labels of the test data. You can use the ```predict()``` function to predict the labels of the test data. You can change the ```mode``` parameter to select which individuals are used to predict.
- "all": uses all individuals (models) from the last generation
- "best": uses the from the last generation which achieved the best predictive metric
- "simplest": uses the from the last generation which achieved the best efficiency metric
- "pareto": uses the pareto individuals from the last generation (only for multiobjective. These are the models that achieved simultaneouly the best predictive metric and efficiency metric.

Additionally, you can use the ```threshold``` parameter (only used for AutoEncoders) to set the threshold for the prediction. You can use the following values:
- "default": uses a different threshold value for each individual (model). For each model the threshold value is the associated default value (currently this works similar to the "mean" value).
- "mean": For each model the threshold value is the sum of the mean reconstruction error obtained on the validation data and one standard deviation.
- "percentile": For each model the threshold value is the 95th percentile of the reconstruction error obtained on the validation data (you can also use the ```percentile``` parameter to change the percentile).
- "max": For each model the threshold value is maximum reconstruction error obtained on the validation data.
- You can also pass an Integer of Float value. In this case, the threshold value is the same for all the models.


```python
predictions = aoc.predict(X_test,
    mode="all",
    threshold="default")
```

### 6. Evaluate

You can use the predictions to calculate manually the performance metrics of the model. However, the ```evaluate()``` function is a more convenient way to do it. You can also use the ```mode``` parameter (works similarly to the ```predict()``` function) and use metrics from the ```sklearn.metrics``` package (currently available are "roc_auc", "accuracy", "precision", "recall", and "f1").

```python
score = aoc.evaluate(X_test,
    y_test,
    mode="all",
    metric="roc_auc",
    threshold="default")
```

## Usage (Full Example)

```python
from autooc.autooc import AutoOC

aoc = AutoOC(anomaly_class = 0,
    normal_class = 1,
    multiobjective=True,
    performance_metric="training_time",
    algorithm = "autoencoder"
)

X_train, X_val, X_test, y_test = aoc.load_example_data()

run = aoc.fit(
    X=X_train,
    X_val=X_val,
    pop=3,
    gen=3,
    epochs=100,
    mlflow_tracking_uri="../results",
    mlflow_experiment_name="test_experiment",
    mlflow_run_name="test_run",
    results_path="../results"
)

predictions = aoc.predict(X_test,
    mode="all",
    threshold="default")

score = aoc.evaluate(X_test,
    y_test,
    mode="all",
    metric="roc_auc",
    threshold="default")
print(score)
```

<!--_For more examples, please refer to the [Documentation](https://example.com)_-->

<!-- CITATION -->
## Citation

To cite this work please use the following article:

```
@article{FERREIRA2023110496,
  author = {Luís Ferreira and Paulo Cortez}
  title = {AutoOC: Automated multi-objective design of deep autoencoders and one-class classifiers using grammatical evolution},
  journal = {Applied Soft Computing},
  volume = {144},
  pages = {110496},
  year = {2023},
  issn = {1568-4946},
  doi = {https://doi.org/10.1016/j.asoc.2023.110496},
  url = {https://www.sciencedirect.com/science/article/pii/S1568494623005148}
}
```

### Built With

* [Python](https://www.python.org)
* [PonyGE2](https://github.com/PonyGE/PonyGE2)
* [TensorFlow](https://www.tensorflow.org/)
* [Scikit-Learn](https://scikit-learn.org/)

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/luisferreira97/AutoOC) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Luís Ferreira - [LinkedIn](https://www.linkedin.com/in/luisferreira97/) - luis_ferreira223@hotmail.com

Project Link: [https://github.com/luisferreira97/AutoOC](https://github.com/luisferreira97/AutoOC)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [PonyGE2](https://github.com/PonyGE/PonyGE2)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/luisferreira97/AutoOC.svg?style=for-the-badge
[contributors-url]: https://github.com/luisferreira97/AutoOC/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/luisferreira97/AutoOC.svg?style=for-the-badge
[forks-url]: https://github.com/luisferreira97/AutoOC/network/members
[stars-shield]: https://img.shields.io/github/stars/luisferreira97/AutoOC.svg?style=for-the-badge
[stars-url]: https://github.com/luisferreira97/AutoOC/stargazers
[issues-shield]: https://img.shields.io/github/issues/luisferreira97/AutoOC.svg?style=for-the-badge
[issues-url]: https://github.com/luisferreira97/AutoOC/issues
[license-shield]: https://img.shields.io/github/license/luisferreira97/AutoOC.svg?style=for-the-badge
[license-url]: https://github.com/luisferreira97/AutoOC/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/luisferreira97/
[product-screenshot]: images/logo.png
