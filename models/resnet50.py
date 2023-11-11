import numpy as np
import tensorflow as tf
from ray import tune
import ray
from models.base_model import BaseModel
from overrides import overrides
from tensorflow.keras import models, layers
from keras.applications import ResNet50


class ResnetFifty(BaseModel):

    def build_model(self, train_X, config):
        # Initialize model
        inputs = layers.Input(shape=train_X.shape[1:])
        
        # Add a padding layer to change the input shape from (28, 28, 1) to (32, 32, 1)
        x = layers.ZeroPadding2D(padding=(2, 2))(inputs)  # Add 2 pixels of padding on each side

        classes = 10

        # Note: You should change the input_shape in ResNet50 to (32, 32, 1)
        outputs = ResNet50(
            weights=None, input_shape=(32, 32, 1), classes=classes
        )(x)

        model = models.Model(inputs, outputs)

        # Define optimizer
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"])
        
        # Compile model
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
                patience=self.configs["patience"],  # Number of epochs with no improvement to wait before stopping
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
            "epochs": tune.choice(self.configs["epochs"]),
            "batch_size": tune.choice(self.configs["batch_size"]),
            "learning_rate": tune.choice(self.configs["learning_rate"]),
        }

        analysis = tune.run(
            self.train,
            config=search_space,
            num_samples=self.configs["num_samples"],
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
