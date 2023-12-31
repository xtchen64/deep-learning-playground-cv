import numpy as np
import tensorflow as tf
from ray import tune
import ray
from models.base_model import BaseModel
from overrides import overrides


class Ffn(BaseModel):

    def build_model(self, train_X, config):
        """
        Builds a simple feed-forward neural network based on the given config.
        For this function, we use the Keras Sequential API.
        :param train_X: training data
        :param config: dictionary of hyperparameters
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=train_X.shape[1:]),
            tf.keras.layers.Dense(config["hidden_units"][0], activation=config["activation"], kernel_initializer=config["initializer"]),
            tf.keras.layers.Dropout(config["dropout_rate"]),
            tf.keras.layers.Dense(config["hidden_units"][1], activation=config["activation"], kernel_initializer=config["initializer"]),
            tf.keras.layers.Dropout(config["dropout_rate"]),
            tf.keras.layers.Dense(config["hidden_units"][2], activation=config["activation"], kernel_initializer=config["initializer"]),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"])
        
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    
    def build_model_functional_api(self, train_X, config):
        """
        Builds a simple feed-forward neural network based on the given config.
        For this function, we use the Keras Functional API.
        This function is not used in the main code, but is included here for reference.
        :param train_X: training data
        :param config: dictionary of hyperparameters
        """
        
        # Define input layer
        inputs = tf.keras.layers.Input(shape=train_X.shape[1:])
        
        # Connect layers
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(config["hidden_units"][0], activation=config["activation"], kernel_initializer=config["initializer"])(x)
        x = tf.keras.layers.Dropout(config["dropout_rate"])(x)
        x = tf.keras.layers.Dense(config["hidden_units"][1], activation=config["activation"], kernel_initializer=config["initializer"])(x)
        x = tf.keras.layers.Dropout(config["dropout_rate"])(x)
        x = tf.keras.layers.Dense(config["hidden_units"][2], activation=config["activation"], kernel_initializer=config["initializer"])(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config["learning_rate"])
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model
    
    def build_model_subclass_api(self, config):
        """
        Builds a simple feed-forward neural network based on the given config.
        For this function, we use the Keras Subclass API.
        This function is not used in the main code, but is included here for reference.
        Unlike other API, for subclass API, we don't need train_X provided as input.
        :param config: dictionary of hyperparameters
        """
        # Create custom model
        model = CustomModel(config)
    
        # Compile the model
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

        # Create model. We provide 3 equivalent APIs to build the model.
        model = self.build_model(train_X, config)
        # model = self.build_model_functional_api(train_X, config)
        # model = self.build_model_subclass_api(config)
        
        if verbose:
            print(f"\nTraining FFN...")

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
            "hidden_units": tune.choice(self.configs["hidden_units"]),
            "activation": tune.choice(self.configs["activation"]),
            "epochs": tune.choice(self.configs["epochs"]),
            "batch_size": tune.choice(self.configs["batch_size"]),
            "learning_rate": tune.choice(self.configs["learning_rate"]),
            "dropout_rate": tune.choice(self.configs["dropout_rate"]),
            "initializer": tune.choice(self.configs["initializer"]),
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
        pred_probs = model.predict(X)
        y_pred = np.argmax(pred_probs, axis=1)
        return y_pred


class CustomModel(tf.keras.Model):
    """
    Custom model using the Keras Subclass API.
    This class is not necessary in the main code, but is included here for reference.
    """
    def __init__(self, config, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        
        # Initialize layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(config["hidden_units"][0], 
                                            activation=config["activation"], 
                                            kernel_initializer=config["initializer"])
        self.dropout1 = tf.keras.layers.Dropout(config["dropout_rate"])
        self.dense2 = tf.keras.layers.Dense(config["hidden_units"][1], 
                                            activation=config["activation"], 
                                            kernel_initializer=config["initializer"])
        self.dropout2 = tf.keras.layers.Dropout(config["dropout_rate"])
        self.dense3 = tf.keras.layers.Dense(config["hidden_units"][2], 
                                            activation=config["activation"], 
                                            kernel_initializer=config["initializer"])
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs, training=False):
        # Forward pass
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        return self.output_layer(x)
