#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def load_data(path):
    # Read CSV (utf-8 by default)
    df = pd.read_csv(path)
    # Basic checks
    if not set(['text','label']).issubset(df.columns):
        raise ValueError("CSV must contain columns: text,label")
    df = df.dropna(subset=['text','label']).copy()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    # Keep only spam/ham
    df = df[df['label'].isin(['spam','ham'])]
    if df.empty:
        raise ValueError("No valid rows with labels 'spam' or 'ham'.")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data/dataset.csv', help='Path to dataset CSV (columns: label,text)')
    ap.add_argument('--model_dir', default='../models', help='Directory to save model & vectorizer')
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    df = load_data(args.data)
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    # Vectorizer: unigrams + bigrams
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=1)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    # Model
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train_vec, y_train)

    # Eval
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='spam', zero_division=0)

    print("=== Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham','spam'], zero_division=0))

    # Save
    joblib.dump(clf, os.path.join(args.model_dir, 'spam_model.pkl'))
    joblib.dump(vec, os.path.join(args.model_dir, 'vectorizer.pkl'))
    print(f"\nSaved model & vectorizer to: {args.model_dir}")

if __name__ == '__main__':
    main()
