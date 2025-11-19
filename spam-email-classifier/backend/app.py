#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
from pathlib import Path

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_PATH = MODEL_DIR / 'spam_model.pkl'
VEC_PATH = MODEL_DIR / 'vectorizer.pkl'

def load_pipeline():
    if not MODEL_PATH.exists() or not VEC_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VEC_PATH)
    return model, vec

@app.route('/')
def index():
    # Serve the frontend
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'ok': False, 'error': 'Empty text'}), 400

    model, vec = load_pipeline()
    if model is None:
        return jsonify({'ok': False, 'error': 'Model not found. Please train first.'}), 500

    X = vec.transform([text])
    pred = model.predict(X)[0]
    return jsonify({'ok': True, 'prediction': pred})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
