import numpy as np
import tensorflow as tf
from ray import tune
import ray
from models.base_model import BaseModel
from overrides import overrides
from tensorflow.keras import models, layers
from tensorflow.keras import regularizers


class Resnet(BaseModel):

    def _residual_block(self, x, filters, kernel_size, padding, activation, batchnorm_momentum, name, l2_penalty):
        # Residual block with two convolutional layers and a skip connection
        input_tensor = x

        # Adjust the number of channels in the input_tensor to match `filters`
        # using a 1x1 convolution if the number of filters is different.
        if input_tensor.shape[-1] != filters:
            input_tensor = layers.Conv2D(filters, (1, 1), padding=padding, name=name+"_conv0", kernel_regularizer=regularizers.l2(l2_penalty))(input_tensor)
            input_tensor = layers.BatchNormalization(momentum=batchnorm_momentum, name=name+"_bn0")(input_tensor)

        x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation, name=name+"_conv1", kernel_regularizer=regularizers.l2(l2_penalty))(x)
        x = layers.BatchNormalization(momentum=batchnorm_momentum, name=name+"_bn1")(x)
        x = layers.Activation(activation)(x)

        x = layers.Conv2D(filters, kernel_size, padding=padding, name=name+"_conv2", kernel_regularizer=regularizers.l2(l2_penalty))(x)
        x = layers.BatchNormalization(momentum=batchnorm_momentum, name=name+"_bn2")(x)

        # Skip Connection
        x = layers.Add(name=name+"_add")([x, input_tensor])
        x = layers.Activation(activation)(x)
        return x

    def build_model(self, train_X, config):
        # Initialize model
        inputs = layers.Input(shape=train_X.shape[1:])
        x = inputs

        # Initial Conv Layer
        x = layers.Conv2D(
            filters=config["num_conv_channels"],
            kernel_size=(config["conv_kernel_size"], config["conv_kernel_size"]),
            padding=config["padding"],
            name="initial_conv",
            kernel_regularizer=regularizers.l2(config["l2_penalty"])
        )(x)
        x = layers.BatchNormalization(momentum=config["batchnorm_momentum"], name="initial_bn")(x)
        x = layers.Activation(config["activation"])(x)

        # Build Residual Block
        for i in range(config["num_res_blocks"]):
            x = self._residual_block(
                x=x, 
                filters=config["num_conv_channels"], # Same number of filters of the initial conv2d layer
                kernel_size=config["conv_kernel_size"],
                padding=config["padding"], 
                activation=config["activation"], 
                batchnorm_momentum=config["batchnorm_momentum"],
                name=f"resblock_{i+1}",
                l2_penalty=config["l2_penalty"]
            )

        # Classifier
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dense(config["hidden_units"], activation=config["activation"], name="dense_1")(x)
        outputs = layers.Dense(10, activation='softmax', name="output")(x)

        model = tf.keras.models.Model(inputs, outputs)

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
            print(f"\nTraining Resnet50...")

        callbacks = []

        # Add TensorBoard callback
        if tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir="/Users/xtchen/Projects/deep-learning-playgrouind-cv/results/tensorboard", 
                histogram_freq=1
            )
            callbacks.append(tensorboard_callback)

        if self.configs["early_stopping"]:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  # or 'val_accuracy'
                patience=10,  # Number of epochs with no improvement to wait before stopping
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
            "num_conv_channels": tune.choice(self.configs["num_conv_channels"]),
            "conv_kernel_size": tune.choice(self.configs["conv_kernel_size"]),
            "padding": tune.choice(self.configs["padding"]),
            "activation": tune.choice(self.configs["activation"]),
            "num_res_blocks": tune.choice(self.configs["num_res_blocks"]),
            "hidden_units": tune.choice(self.configs["hidden_units"]),
            "epochs": tune.choice(self.configs["epochs"]),
            "batch_size": tune.choice(self.configs["batch_size"]),
            "learning_rate": tune.choice(self.configs["learning_rate"]),
            "initializer": tune.choice(self.configs["initializer"]),
            "batchnorm_momentum": tune.choice(self.configs["batchnorm_momentum"]),
            "l2_penalty": tune.choice(self.configs["l2_penalty"]),
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
