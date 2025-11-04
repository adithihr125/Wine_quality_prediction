from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open('wine_quality_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)

    result = 'Good Quality Wine' if prediction[0] == 1 else 'Bad Quality Wine'
    return jsonify({'prediction': int(prediction[0]), 'result': result})

if __name__ == '__main__':
    app.run(debug=True)
