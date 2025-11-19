#!/usr/bin/env python3
import joblib
from pathlib import Path

def load_pipeline(model_dir='../models'):
    model = joblib.load(Path(model_dir) / 'spam_model.pkl')
    vec = joblib.load(Path(model_dir) / 'vectorizer.pkl')
    return model, vec

def predict_label(text, model_dir='../models'):
    model, vec = load_pipeline(model_dir)
    X = vec.transform([text])
    return model.predict(X)[0]

if __name__ == '__main__':
    # simple manual test
    print(predict_label("Win a FREE iPhone now! Click here", model_dir='../models'))
