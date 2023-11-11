"""
This file contains the setup for the project.
To setup environment, run `pip install -e .` in terminal.
"""
from setuptools import setup, find_packages

setup(
    name="deep-learning-playgrouind-cv",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'gin-config',
        'tensorflow==2.15.0rc0',
        'tensorflow-metal',
        'ray[tune]',
        'ipywidgets',
        'overrides'
    ]
)
