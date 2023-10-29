import gin
from models.logistic_reg import LogisticReg
from data_processors.base_data_processor import BaseDataProcessor
from utils.common_utils import write_results_to_file

@gin.configurable
def main(configs, write_results=True):
    # TODO: add more configs as gin configurable (e.g. computation budget)
    # TODO: configure GPU for ray tune (e.g. might need to use specific packages)

    # process data
    data_processor = BaseDataProcessor()
    data_processor.setup()
    train_X, train_y, val_X, val_y, test_X, test_y = data_processor.load_data()

    # initialize model
    lr = LogisticReg(
        train_X,
        train_y,
        val_X,
        val_y,
        configs,
    )

    # train model with ray tune
    best_lr_metrics = lr.train_with_ray_tune()

    # evaluate model on training set
    print("\nBest model training performance:")
    acc, f1 = lr.evaluate(lr.model, train_X, train_y)
    print("Train acc: {}".format(acc))
    print("Train f1: {}".format(f1))

    # evaluate model on test set
    print("\nBest model test performance:")
    acc, f1 = lr.evaluate(lr.model, test_X, test_y)
    print("Test acc: {}".format(acc))
    print("Test f1: {}".format(f1))

    # write result to csv
    if write_results:
        write_results_to_file(
            acc, 
            f1, 
            model_name="logistic_reg", 
            filename="/Users/xtchen/Projects/deep-learning-with-mnist/result.csv")

if __name__ == "__main__":
    gin.parse_config_file('pipelines/configs/logistic_reg_pipeline.gin')
    main()
