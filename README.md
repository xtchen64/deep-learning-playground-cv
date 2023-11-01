# deep-learning-with-mnist
Deep Learning with MNIST dataset

## Set up:
Note that the environment does not use brew or conda.
1. Set up local virtual environment (vscode only). This sets up a venv in the project folder:
    - In vscode terminal, make sure you are in the project folder. Create venv by running `python -m venv mnist`
    - In vscode, if not prompted to do so, run `command+shift+p`, choose the `mnist` interpreter
    - Activate this venv in terminal: `source mnist/bin/activate`
2. Install required packages in terminal: `pip install -e .`

## Train and evaluate models:
1. To run pipeline in terminal, configure with gin files and run commands such as `python pipelines/flows/logistic_reg_pipeline.py`. 
Alternatively, experiment with the code in the example notebooks in the `notebooks` folder.
2. For tensorflow models with tensorboard configured, run command such as `tensorboard --logdir results/tensorboard` to view tensorboard, where `results/tensorboard` is the folder for the logs.

## Supported Models:
1. Logistic Regression
2. Feed Forward Network (FFN)
3. [WIP] Convoluted Neural Network (CNN)