import numpy as np
import tensorflow as tf
from ray import tune
import ray
from models.base_model import BaseModel
from overrides import overrides
from tensorflow.keras import models, layers


class Cnn(BaseModel):

    def build_model(self, train_X, config):
        """
        Builds a simple convoluted neural network based on the given config.
        :param train_X: training data
        :param config: dictionary of hyperparameters

        Example config: 
        {
            "conv_channels": [
                [16, 32, 32], 
                [32, 64, 64], 
            ],  # Different possible number of channels for the 3 conv layers
            "conv_kernel_size": [3, 5],  # Different possible kernel sizes for each conv layer
            "padding": ['same', 'valid'], # Different possible padding options
            "activation": ['relu', 'tanh'],      # Different possible activation functions
            "hidden_units": [32, 64],  # Different possible number of hidden units
            "epochs": [20],                         # Different possible epochs 
            "batch_size": [64, 128],                    # Different possible batch sizes
            "learning_rate": [1e-5, 1e-4, 1e-3],            # Different possible learning rates
            "initializer": ['he_uniform'], # Different possible initializers
            "batchnorm_momentum": [0.9, 0.99],               # Different possible batchnorm momentum
            "early_stopping": [True]
        }
        """
        # Initialize model
        model = models.Sequential()  # Input shape: (28, 28)
        pad = config["padding"]
        activation = config["activation"]
        kernel_size = config["conv_kernel_size"]

        # Add CNN layers: 3 Conv2D layers and 2 MaxPooling2D layers
        model.add(layers.Conv2D(
            config["conv_channels"][0], 
            (kernel_size, kernel_size), 
            activation=config["activation"], 
            input_shape=train_X.shape[1:],
            padding=pad,
            name="conv_1"
        ))  # 1st conv layer
        model.add(layers.MaxPooling2D((2, 2), padding=pad, name="maxpool_1")) # 1st max pooling layer
        model.add(layers.Conv2D(
            config["conv_channels"][1], (kernel_size, kernel_size), activation=activation, padding=pad, name="conv_2"
        ))  # 2nd conv layer
        model.add(layers.MaxPooling2D((2, 2), padding=pad, name="maxpool_2")) # 2nd max pooling layer
        model.add(layers.Conv2D(
            config["conv_channels"][2], (kernel_size, kernel_size), activation=activation, padding=pad, name="conv_3"
        ))  # 3rd conv layer

        # Add Dense layers
        model.add(layers.Flatten(name="flatten"))  # Flatten output of conv layers
        model.add(layers.Dense(config["hidden_units"], activation=activation, name="dense_1"))  # hidden dense layer
        model.add(layers.Dense(10, activation='softmax', name="output"))  # Output layer
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"])
        
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    @overrides
    def train(self, config, verbose=False, tensorboard=False):
        # Get the actual data from the Ray object store
        train_X = ray.get(self.train_X_ref)
        train_y = ray.get(self.train_y_ref)
        val_X = ray.get(self.val_X_ref)
        val_y = ray.get(self.val_y_ref)

        # Add a new dimension to the data: (28, 28) -> (28, 28, 1)
        train_X = np.expand_dims(train_X, axis=-1)
        val_X = np.expand_dims(val_X, axis=-1)

        # Create model. We provide 3 equivalent APIs to build the model.
        model = self.build_model(train_X, config)
        # model = self.build_model_functional_api(train_X, config)
        # model = self.build_model_subclass_api(config)
        
        if verbose:
            print(f"\nTraining CNN...")

        callbacks = []

        # Add TensorBoard callback
        if tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir="/Users/xtchen/Projects/deep-learning-with-mnist/results/tensorboard", 
                histogram_freq=1
            )
            callbacks.append(tensorboard_callback)

        if self.configs["early_stopping"]:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  # or 'val_accuracy'
                patience=2,  # Number of epochs with no improvement to wait before stopping
                verbose=1,
                restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
            )
            callbacks.append(early_stopping)
        
        model.fit(
            train_X, 
            train_y, 
            validation_data=(val_X, val_y), 
            callbacks=callbacks,
            epochs=config["epochs"], 
            batch_size=config["batch_size"], 
            verbose=verbose
        )

        # Evaluate model
        val_acc, val_f1 = self.evaluate(model, val_X, val_y)
        if verbose:
            print("\nValidation Dataset Performance:")
            print(f"Validation accuracy: {val_acc}")
            print(f"Validation F1: {val_f1}")
            
        self.model = model
        
        # Return a dictionary with metrics
        metrics = {
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        }
        
        return metrics

    @overrides
    def train_with_ray_tune(self):
        print("\nStart Ray Tune hyperparameter search...")
        search_space = {
            "conv_channels": tune.choice(self.configs["conv_channels"]),
            "conv_kernel_size": tune.choice(self.configs["conv_kernel_size"]),
            "padding": tune.choice(self.configs["padding"]),
            "activation": tune.choice(self.configs["activation"]),
            "hidden_units": tune.choice(self.configs["hidden_units"]),
            "epochs": tune.choice(self.configs["epochs"]),
            "batch_size": tune.choice(self.configs["batch_size"]),
            "learning_rate": tune.choice(self.configs["learning_rate"]),
            "initializer": tune.choice(self.configs["initializer"]),
            "batchnorm_momentum": tune.choice(self.configs["batchnorm_momentum"]),
            "early_stopping": tune.choice(self.configs["early_stopping"])
        }

        analysis = tune.run(
            self.train,
            config=search_space,
            num_samples=20,
            resources_per_trial={"cpu": 10, "gpu": 0},
        )

        best_config = analysis.get_best_config(metric="val_accuracy", mode="max")
        print("Best hyperparameters:", best_config)

        # Train the model with the best hyperparameters
        self.train(best_config)

        print("Training completed with the best hyperparameters.")

    @overrides
    def predict(self, model, X):
        """
        Predicts on the given data. Can take in 3d or 4d arrays.
        """
        # if X has 3 dimensions, then we need to add one dimension to make it 4 dimensions
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        pred_probs = model.predict(X)
        y_pred = np.argmax(pred_probs, axis=1)
        return y_pred
