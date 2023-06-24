
# read the contents of your README file
from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="talos-automl",
    version="0.0.4",
    author_email="luis_ferreira223@hotmail.com",
    author="Luís Ferreira",
    description="Talos: Automated Machine Learning (AutoML) library for One-Class Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/luisferreira97/talos",
    License="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=["talos"],
    python_requires='>=3.6',
    keywords=['automl', 'machine learning', 'one-class learning',
              'one-class classification', 'autoencoder', 'isolation forest', 'one-class svm'],
    include_package_data=True,
    #package_dir={"": "src/talos"},
    install_requires=[
        'keras==2.6.0',
        'matplotlib==3.4.1',
        'mlflow==1.15.0',
        'pandas==1.2.4',
        'pre-commit==2.15.0',
        'scikit-learn==1.0',
        'tensorflow==2.6.0',
        'tensorflow-estimator==2.6.0',
        'tqdm==4.60.0'
    ]
)
