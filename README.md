:

🛡️ Phishing Email & URL Detector (DistilBERT-powered)
This Streamlit web app uses a fine-tuned DistilBERT model to detect phishing attempts in both email text and URLs. Users can enter content manually or upload .txt or .csv files, and receive predictions with confidence scores.

🔍 Features
🧠 DistilBERT NLP model trained on phishing email and URL data

📥 Accepts:

Manual text input

.txt files (one message per line)

.csv files (must contain a text column)

👤 Users can:

Classify phishing content

View their own prediction history

🔐 Admin can:

View logs from all users

Access protected with a password stored in .env or Streamlit secrets

🚀 Technologies
Python · Streamlit · Transformers · Hugging Face · Torch · Pandas · Dotenv

