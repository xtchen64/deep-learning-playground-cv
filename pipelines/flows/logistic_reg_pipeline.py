import gin
from models.logistic_reg import LogisticReg
from data_processors.base_data_processor import BaseDataProcessor

@gin.configurable
def main(penalty, C, random_state, max_iter):

    # process data
    data_processor = BaseDataProcessor()
    data_processor.setup()
    train_X, train_y, val_X, val_y, test_X, test_y = data_processor.load_data()

    # initialize model
    model = LogisticReg(penalty, C, random_state, max_iter)

    # train model
    model.train(train_X, train_y, val_X, val_y)

    # evaluate model
    acc, f1 = model.evaluate(test_X, test_y)
    print("\nTest Dataset Performance:")
    print("Test acc: {}".format(acc))
    print("Test f1: {}".format(f1))

if __name__ == "__main__":
    gin.parse_config_file('pipelines/configs/logistic_reg_pipeline.gin')
    main()
