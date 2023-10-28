import gin
from models.logistic_reg import LogisticRegression
from utils.common_utils import *
from data_processors import BaseDataProcessor

@gin.configurable
def main(learning_rate, epochs):
    # process data
    data_processor = BaseDataProcessor()
    train_X, train_y, val_X, val_y, test_X, test_y = data_processor.load_data()

    # train model
    model = LogisticRegression(learning_rate, epochs)
    model.train(train_X, train_y, val_X, val_y)

    # evaluate model
    model.evaluate(test_X, test_y)

    # TODO: save model and eval results

if __name__ == "__main__":
    gin.parse_config_file('pipelines/configs/logistic_reg_pipeline.gin')
    main()
