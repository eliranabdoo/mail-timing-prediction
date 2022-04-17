import dill

from utils import predict_hours, load_train_data

MODEL_PATH = r"C:\Users\darkr\PycharmProjects\rightbound\outputs\2022-04-18\00-51-01\model.pickle"


def load_data():
    X = load_train_data(*["C:/Users/darkr/PycharmProjects/rightbound/data/eng.csv",
                          "C:/Users/darkr/PycharmProjects/rightbound/data/companies.csv",
                          "C:/Users/darkr/PycharmProjects/rightbound/data/contacts.csv"]).data
    return X


def main():
    x = load_data().iloc[0:1, :]
    with open(MODEL_PATH, 'rb') as f:
        model = dill.load(f)
    hours = predict_hours(model, x)
    print(hours)


if __name__ == "__main__":
    main()
