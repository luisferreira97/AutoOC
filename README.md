<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!--[![Downloads](https://static.pepy.tech/personalized-badge/talos-automl?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/talos-automl)-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/luisferreira97/talos">
    <img src="images/logo.png" alt="Logo" width="80" height="115">
  </a>

  <h3 align="center">Talos (in development)</h3>

  <p align="center">
    Talos: Automated Machine Learning (AutoML) library focused on One-Class Learning algorithms (AutoEncoders, Isolation Forest and One-Class SVM)
    <br />
    <a href="https://github.com/luisferreira97/talos"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/luisferreira97/talos">View Demo</a>
    ·
    <a href="https://github.com/luisferreira97/talos/issues">Report Bug</a>
    ·
    <a href="https://github.com/luisferreira97/talos/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
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
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

There are many great README templates available on GitHub, however, I didn't find one that really suit my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should element DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have have contributed to expanding this template!

A list of commonly used resources that I find helpful are listed in the acknowledgements.

### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Python](https://www.python.org)
* [PonyGE2](https://github.com/PonyGE/PonyGE2)
* [TensorFlow](https://www.tensorflow.org/)
* [Scikit-Learn](https://scikit-learn.org/)



<!-- GETTING STARTED -->
## Getting Started

This section presents how the package can be reached and installed.

### Where to get it

The source code is currently hosted on GitHub at: https://github.com/luisferreira97/talos

Binary installer for the latest released version are available at the Python Package Index (PyPI).
Note that the PyPI name of the package is `talos-automl` and not `talos`.

```sh
pip install talos-automl
```

<!-- USAGE EXAMPLES -->
## Usage

### 1. Import the package
The first step in using the package is, after it has been installed, to import it. The main class from which all the methods are available is ```Talos```.

```python
from talos.talos import Talos
```

### 2. Instantiate a Talos object
The second step is to instantiate the Talos class with the information about your dataset and context (e.g., normal and anomaly classes, wether to run single-objective or multi-objective, the performance_metric, and the algorithm).
You can change the ```algorithm``` parameter to select which algorithms are used during the optimization. The options are:
- "autoencoders": Deep AutoEncoders (from TensorFlow)
- "iforest": Isolation Forest (from Scikit-Learn)
- "svm": One-Class SVM (from Scikit-Learn)
- "all": the optimization is done using the three algorithms above

```python
talos = Talos(anomaly_class = 0,
    normal_class = 1,
    multiobjective=True,
    performance_metric="training_time",
    algorithm = "autoencoder"
)
```

### 3. Load dataset
The third step is to load the dataset. Depending on the type of validation you need *train data* (only 'normal' instances), *validation data* (you can use (1) only 'normal' instances or (2) both 'normal' and 'anomaly' instances with the respective labels), and *test data* (both types of instances and labels). You can use the ```load_example_data()``` function to load the popular ECG dataset.


```python
X_train, X_val, X_test, y_test = talos.load_example_data()
```

### 4. Train
The fourth step is to train the model. The ```fit()``` function computes the optimization using the given parameters.

```python
run = talos.fit(
    X=X_train,
    X_val=X_val,
    pop=10,
    gen=10,
    experiment_name="test",
    epochs=1000
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
predictions = talos.predict(X_test,
    mode="all",
    threshold="default")
```

### 6. Evaluate

You can use the predictions to calculate manually the performance metrics of the model. However, the ```evaluate()``` function is a more convenient way to do it. You can also use the ```mode``` parameter (works similarly to the ```predict()``` function) and use metrics from the ```sklearn.metrics``` package (currently available are "roc_auc", "accuracy", "precision", "recall", and "f1").

```python
score = talos.evaluate(X_test,
    y_test,
    mode="all",
    metric="roc_auc",
    threshold="default")
```

## Usage

```python
from talos.talos import Talos

talos = Talos(anomaly_class = 0,
    normal_class = 1,
    multiobjective=True,
    performance_metric="training_time",
    algorithm = "autoencoder"
)

X_train, X_val, X_test, y_test = talos.load_example_data()

run = talos.fit(
    X=X_train,
    X_val=X_val,
    pop=10,
    gen=10,
    experiment_name="test",
    epochs=1000
)

predictions = talos.predict(X_test,
    mode="all",
    threshold="default")

score = talos.evaluate(X_test,
    y_test,
    mode="all",
    metric="roc_auc",
    threshold="default")
```

_For more examples, please refer to the [Documentation](https://example.com)_-->



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/luisferreira97/talos) for a list of proposed features (and known issues).



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

Luís Ferreira - [LinkedIn](https://www.linkedin.com/in/luisferreira97/) - email@example.com

Project Link: [https://github.com/luisferreira97/talos](https://github.com/luisferreira97/talos)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [PonyGE2](https://github.com/PonyGE/PonyGE2)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/luisferreira97/talos/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/luisferreira97/talos/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/luisferreira97/talos/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/luisferreira97/talos/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/luisferreira97/talos/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/luisferreira97/
[product-screenshot]: images/screenshot.png
