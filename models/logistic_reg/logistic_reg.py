import numpy as np

from utils.common_utils import auto_assign

class LogisticReg():
    @auto_assign
    def __init__(
        self, 
        learning_rate=0.01, 
        epochs=100,
    ):
        pass
    
    def train(self, data):
        # Logic for training
        print(f"Training logistic reg with learning_rate: {self.learning_rate} for {self.epochs} epochs.")
