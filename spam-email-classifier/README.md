# Spam Email Classifier

A complete project to classify emails as **spam** or **ham** using **Python**, **Flask**, and **Machine Learning** (TF-IDF + Multinomial Naive Bayes). This project includes a backend API and a frontend interface.

---

## Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/rishieeee/spam-email-classifier.git
cd spam-email-classifier

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

2️⃣ Create and activate a virtual environment

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate


3️⃣ Install required packages

pip install -r requirements.txt

Prepare the Dataset

Place a CSV file at data/dataset.csv.

Required columns:

text – Email content

label – Either spam or ham

Example dataset:
label,text
ham,"Hey! Are we still meeting for lunch today?"
spam,"Win a brand new car! Click here to claim your prize now."


Train the Model
From the backend folder, run:

cd backend
python train.py --data ../data/dataset.csv --model_dir models

This will:

Split the dataset into train/test sets

Train a TF-IDF + Multinomial Naive Bayes pipeline

Save the trained model to backend/models/spam_model.pkl

Run the Backend API

cd backend
python app.py

The Flask API will run at: http://127.0.0.1:5000/

Endpoints:

/ → Serves the frontend page (index.html)

/api/predict → POST endpoint for spam/ham prediction

Example POST request (JSON):
{
  "text": "Win a FREE iPhone now! Click here"
}

Example response:
{
  "ok": true,
  "prediction": "Spam"
}

Project Structure

spam-email-classifier/
│
├── backend/
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   └── models/
│       ├── spam_model.pkl
│       └── vectorizer.pkl
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── data/
│   └── dataset.csv
├── .venv/
├── requirements.txt
└── README.md


Notes & Tips

Ensure your Python version matches the one used during training to avoid sklearn compatibility issues

For larger datasets, consider using class weights or other classifiers like Logistic Regression or Linear SVM

Use pip install --upgrade scikit-learn pandas if encountering version mismatch warnings



