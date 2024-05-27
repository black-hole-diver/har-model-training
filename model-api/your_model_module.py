import joblib

class YourModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions