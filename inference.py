import dill

from utils import predict_hours, load_train_data

MODEL_PATH = r".\outputs\2022-04-18\00-51-01\model.pickle"


def load_data():
    # Don't really use the train data...
    X = load_train_data(*["./data/eng.csv",
                          "./data/companies.csv",
                          ".data/contacts.csv"]).data
    return X


def main():
    x = load_data()
    with open(MODEL_PATH, 'rb') as f:
        model = dill.load(f)
    hours = predict_hours(model, x)
    print(hours)


if __name__ == "__main__":
    main()
