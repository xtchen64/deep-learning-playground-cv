from keras.datasets import mnist
from sklearn.model_selection import train_test_split


class BaseDataProcessor:
    def __init__(self):
        pass
    
    def setup(self):
        # load mnist datasets
        print("Loading MNIST datasets...")
        (self.train_val_X, self.train_val_y), (self.test_X, self.test_y) = mnist.load_data()
        
        # normalize train_val_X and test_X
        print("Normalizing datasets...")
        self.train_val_X = self.train_val_X / 255.0
        self.test_X = self.test_X / 255.0

        # further split train_val into train and val
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
            self.train_val_X, self.train_val_y, test_size=0.3, random_state=0
        )

        # print sizes of datasets
        print("\nDataset shapes:")
        print('train_X: ' + str(self.train_X.shape))
        print('train_y: ' + str(self.train_y.shape))
        print('val_X: ' + str(self.val_X.shape))
        print('val_y: ' + str(self.val_y.shape))
        print('test_X:  '  + str(self.test_X.shape))
        print('test_y:  '  + str(self.test_y.shape))
    
    def load_data(self):
        return self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y
