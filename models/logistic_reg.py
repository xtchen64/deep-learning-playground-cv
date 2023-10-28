from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class LogisticReg():
    def __init__(
        self, 
        penalty='l2',
        C=1.0,
        random_state=0,
        max_iter=100,
    ):
        # initialize model
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            random_state=random_state,
            max_iter=max_iter,
        )
    
    def train(self, train_X, train_y, val_X, val_y):
        """
        Trains the model on the given data.
        :param train_X: 3d np array to train on (num_samples, height, width)
        :param train_y: 2d array of true labels (num_samples,)
        :param val_X: 3d np array to validate on (num_samples, height, width)
        :param val_y: 2d array of true labels (num_samples,)
        """
        # flatten data
        train_X_flat = train_X.reshape(train_X.shape[0], -1)

        print(f"\nTraining logistic reg...")
        self.model.fit(train_X_flat, train_y)

        # report training performance
        val_acc, val_f1 = self.evaluate(train_X, train_y)
        print("\nTraining Dataset Performance:")
        print(f"Training accuracy: {val_acc}")
        print(f"Training F1: {val_f1}")
        
        # report validation performance
        val_acc, val_f1 = self.evaluate(val_X, val_y)
        print("\nValidation Dataset Performance:")
        print(f"Validation accuracy: {val_acc}")
        print(f"Validation F1: {val_f1}")
    
    def predict(self, X):
        """
        Predicts the labels for the given data.
        :param X: 3d np array to predict on (num_samples, height, width)
        """
        X_flat = X.reshape(X.shape[0], -1)
        y_pred = self.model.predict(X_flat)
        return y_pred

    def evaluate(self, X, y):
        """
        Evaluates the model on the given data.
        :param X: 3d np array to predict on (num_samples, height, width)
        :param y: 2d array of true labels (num_samples,)
        """
        # Predict using the trained model
        predictions = self.predict(X)

        # Calculate accuracy
        accuracy = accuracy_score(y, predictions)

        # Calculate Macro-Averaged F1-score
        f1_macro = f1_score(y, predictions, average='macro')

        return accuracy, f1_macro
