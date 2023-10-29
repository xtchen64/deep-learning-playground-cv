import numpy as np
import tensorflow as tf
from ray import tune
import ray
from sklearn.metrics import accuracy_score, f1_score

class Ffn():
    def __init__(self, train_X, train_y, val_X, val_y, configs):
        self.configs = configs

        # Put large datasets in the Ray object store
        self.train_X_ref = ray.put(train_X)
        self.train_y_ref = ray.put(train_y)
        self.val_X_ref = ray.put(val_X)
        self.val_y_ref = ray.put(val_y)

    def build_model(self, train_X, config):
        """Builds a simple feed-forward neural network based on the given config."""
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

    def train(self, config, verbose=False):
        # Get the actual data from the Ray object store
        train_X = ray.get(self.train_X_ref)
        train_y = ray.get(self.train_y_ref)
        val_X = ray.get(self.val_X_ref)
        val_y = ray.get(self.val_y_ref)

        # Create model
        model = self.build_model(train_X, config)
        
        if verbose:
            print(f"\nTraining FFN...")

        if self.configs["early_stopping"]:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  # or 'val_accuracy'
                patience=2,  # Number of epochs with no improvement to wait before stopping
                verbose=1,
                restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
            )
            callbacks = [early_stopping]
        else:
            callbacks = None
        
        model.fit(
            train_X, 
            train_y, 
            validation_data=(val_X, val_y), 
            callbacks=callbacks,
            epochs=config["epochs"], 
            batch_size=config["batch_size"], 
            verbose=0
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

    def train_with_ray_tune(self):
        print("\nStart Ray Tune hyperparameter search...")
        search_space = {
            "hidden_units": tune.choice(self.configs["hidden_units"]),
            "activation": tune.choice(self.configs["activation"]),
            "epochs": tune.choice(self.configs["epochs"]),
            "batch_size": tune.choice(self.configs["batch_size"]),
            "learning_rate": tune.choice(self.configs["learning_rate"]),
            "dropout_rate": tune.choice(self.configs["dropout_rate"]),
            "initializer": tune.choice(self.configs["initializer"])
        }

        analysis = tune.run(
            self.train,
            config=search_space,
            num_samples=3,
            resources_per_trial={"cpu": 5, "gpu": 0},
        )

        best_config = analysis.get_best_config(metric="val_accuracy", mode="max")
        print("Best hyperparameters:", best_config)

        # Train the model with the best hyperparameters
        self.train(best_config)

        print("Training completed with the best hyperparameters.")

    def predict(self, model, X):
        pred_probs = model.predict(X)
        y_pred = np.argmax(pred_probs, axis=1)
        return y_pred

    def evaluate(self, model, X, y):
        predictions = self.predict(model, X)
        accuracy = accuracy_score(y, predictions)
        f1_macro = f1_score(y, predictions, average='macro')
        return accuracy, f1_macro
