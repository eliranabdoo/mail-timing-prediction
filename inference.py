import dill

from utils import predict_hours

MODEL_PATH = ""


def load_data():
    pass


def main():
    x = load_data()
    with open(MODEL_PATH, 'rb') as f:
        model = dill.load(f)
    hours = predict_hours(model, x)
    print(hours)


if __name__ == "__main__":
    main()
