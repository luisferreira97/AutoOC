
# read the contents of your README file
from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="autooc",
    version="0.0.13",
    author_email="luis_ferreira223@hotmail.com",
    author="Luís Ferreira",
    description="AutoOC: Automated Machine Learning (AutoML) library for One-Class Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/luisferreira97/AutoOC",
    License="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=["autooc"],
    python_requires='>=3.6, <3.9',
    keywords=['automl', 'machine learning', 'one-class learning',
              'one-class classification', 'autoencoder', 'isolation forest', 'one-class svm'],
    include_package_data=True,
    #package_dir={"": "src/autooc"},
    install_requires=[
        'keras==2.6.0',
        'matplotlib==3.4.1',
        'mlflow==1.15.0',
        'pandas==1.2.4',
        'pydot==1.4.2',
        'numpy==1.19.2',
        'scikit-learn==1.0',
        'tensorflow==2.6.0',
        'tensorflow-estimator==2.6.0',
        'tqdm==4.60.0'
    ]
)