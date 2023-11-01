import ray
from sklearn.metrics import accuracy_score, f1_score
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all models.
    """
    def __init__(self, train_X, train_y, val_X, val_y, configs):
        self.configs = configs

        # Put large datasets in the Ray object store
        self.train_X_ref = ray.put(train_X)
        self.train_y_ref = ray.put(train_y)
        self.val_X_ref = ray.put(val_X)
        self.val_y_ref = ray.put(val_y)

    @abstractmethod
    def train(self, config, verbose=False):
        pass

    @abstractmethod
    def train_with_ray_tune(self):
        pass

    @abstractmethod
    def predict(self, model, X):
        pass
    
    def evaluate(self, model, X, y):
        """
        Evaluates the model on the given data.
        :param model: trained model
        :param X: 3d np array to predict on (num_samples, height, width)
        :param y: 2d array of true labels (num_samples,)
        """
        # Predict using the trained model
        predictions = self.predict(model, X)

        # Calculate accuracy
        accuracy = accuracy_score(y, predictions)

        # Calculate Macro-Averaged F1-score
        f1_macro = f1_score(y, predictions, average='macro')

        return accuracy, f1_macro
