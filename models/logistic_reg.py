from ray import tune
import ray
from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel
from overrides import overrides


class LogisticReg(BaseModel):

    @overrides
    def train(self, config, verbose=False):
        """
        Trains the model on the given data. Returns the trained model.
        :param config: dictionary of hyperparameters

        Returns:
        model: trained model
        """
        # Get the actual data from the Ray object store
        train_X = ray.get(self.train_X_ref)
        train_y = ray.get(self.train_y_ref)
        val_X = ray.get(self.val_X_ref)
        val_y = ray.get(self.val_y_ref)

        # Create model
        model = LogisticRegression(
            penalty=config["penalty"], 
            C=config["C"],
            random_state=config["random_state"],
            max_iter=config["max_iter"],
        )

        # flatten data
        train_X_flat = train_X.reshape(train_X.shape[0], -1)

        if verbose:
            print(f"\nTraining logistic reg...")
        
        model.fit(train_X_flat, train_y)
        self.model = model

        # report validation performance
        val_acc, val_f1 = self.evaluate(model, val_X, val_y)
        if verbose:
            print("\nValidation Dataset Performance:")
            print(f"Validation accuracy: {val_acc}")
            print(f"Validation F1: {val_f1}")

        # Return a dictionary with metrics
        metrics = {
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        }
        
        return metrics

    @overrides
    def train_with_ray_tune(self):
        """
        Trains the model on the given data using ray tune. Returns the best model.
        """
        print("\nStart Ray Tune hyperparameter search...")
        search_space = {
            "penalty": tune.choice(self.configs["penalty"]),
            "C": tune.choice(self.configs["C"]),
            "random_state": tune.choice(self.configs["random_state"]),
            "max_iter": tune.choice(self.configs["max_iter"]),
        }

        analysis = tune.run(
            self.train,
            config=search_space,
            num_samples=10,  # You can adjust this to specify how many hyperparameter combinations to try.
            resources_per_trial={"cpu": 5, "gpu": 0},  # Adjust based on your available resources.
        )

        best_config = analysis.get_best_config(metric="val_accuracy", mode="max")
        print("Best hyperparameters:", best_config)

        # Train the model with the best hyperparameters
        self.train(best_config)

        print("Training completed with the best hyperparameters.")
    
    @overrides
    def predict(self, model, X):
        """
        Predicts the labels for the given data.
        :param model: trained model
        :param X: 3d np array to predict on (num_samples, height, width)
        """
        X_flat = X.reshape(X.shape[0], -1)
        y_pred = model.predict(X_flat)
        return y_pred
