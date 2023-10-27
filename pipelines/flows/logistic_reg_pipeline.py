import gin
from models.logistic_reg import LogisticRegression

@gin.configurable
def main(learning_rate, epochs):
    model = LogisticRegression(learning_rate, epochs)
    data = None # Load data here
    model.train(data)

if __name__ == "__main__":
    gin.parse_config_file('config.gin')
    main()