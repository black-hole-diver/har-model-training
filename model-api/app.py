from flask import Flask, request, jsonify
from your_model_module import YourModel

app = Flask(__name__)
model = YourModel('model_weights.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming you're sending JSON data
    predictions = model.predict(data)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)