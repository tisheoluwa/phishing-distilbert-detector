import streamlit as st
import pandas as pd
import torch
import os
from datetime import datetime
from dotenv import load_dotenv
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load environment variables
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Load model and tokenizer from Hugging Face Hub
MODEL_PATH = "tisheoluwa/phishing-distilbert"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# --- Utility Functions ---
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return ("Phishing" if pred == 1 else "Legitimate"), float(probs[0][pred])

def log_prediction(username, text, label, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", f"{username}.csv")
    entry = pd.DataFrame([[timestamp, text, label, confidence]],
                         columns=["timestamp", "text", "label", "confidence"])
    if os.path.exists(filepath):
        entry.to_csv(filepath, mode="a", header=False, index=False)
    else:
        entry.to_csv(filepath, index=False)

def load_user_logs(username):
    path = os.path.join("logs", f"{username}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None

def load_all_logs():
    data = {}
    os.makedirs("logs", exist_ok=True)
    for file in os.listdir("logs"):
        if file.endswith(".csv"):
            user = file.replace(".csv", "")
            df = pd.read_csv(os.path.join("logs", file))
            data[user] = df
    return data

# --- App Layout ---
st.set_page_config(page_title="Phishing Email & URL Detector", layout="centered")
st.title("🛡️ Phishing Email & URL Detector")
st.markdown("Use NLP to detect whether an email or URL is **phishing** or **legitimate**.")

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Predict", "📄 View Logs"])

# --- Predict Tab ---
with tab1:
    st.header("👤 Enter your username")
    username = st.text_input("Username (for saving your history):")

    st.markdown("📩 **Paste email or URL content here:**")
    input_text = st.text_area("", height=200)

    st.markdown("📎 **Or upload a `.txt` or `.csv` file:**")
    file = st.file_uploader("", type=["txt", "csv"])

    if st.button("🔍 Detect"):
        if not username:
            st.warning("Please enter a username.")
        else:
            texts = []

            if input_text:
                texts.append(input_text)

            if file:
                if file.name.endswith(".txt"):
                    file_content = file.read().decode("utf-8")
                    texts.append(file_content.strip())
                elif file.name.endswith(".csv"):
                    try:
                        df = pd.read_csv(file)
                        if "text" in df.columns:
                            texts.extend(df["text"].dropna().tolist())
                        else:
                            st.error("CSV must contain a 'text' column.")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")

            if texts:
                results = []
                for txt in texts:
                    label, conf = classify_text(txt)
                    results.append((txt, label, f"{conf * 100:.2f}%"))
                    log_prediction(username, txt, label, conf)

                result_df = pd.DataFrame(results, columns=["Text", "Prediction", "Confidence"])
                st.success("Detection complete.")
                st.dataframe(result_df)

                # --- Summary Chart ---
                summary = pd.DataFrame(result_df["Prediction"].value_counts()).reset_index()
                summary.columns = ["Prediction", "Count"]
                st.subheader("📊 Prediction Summary")
                st.bar_chart(summary.set_index("Prediction"))

                # --- CSV Export ---
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv, file_name="detection_results.csv", mime="text/csv")
            else:
                st.warning("No valid input provided.")

# --- Logs Tab ---
with tab2:
    st.header("📘 View Logs")
    admin_mode = st.checkbox("I am admin")

    if admin_mode:
        admin_pwd = st.text_input("Enter admin password:", type="password")
        if st.button("Authenticate"):
            if admin_pwd == ADMIN_PASSWORD:
                st.success("Admin authenticated. Displaying all user logs.")
                all_logs = load_all_logs()
                for user, log_df in all_logs.items():
                    st.subheader(f"User: {user}")
                    st.dataframe(log_df)
            else:
                st.error("Invalid admin password.")
    else:
        user = st.text_input("Enter your username to view your log:")
        if st.button("View My Logs"):
            logs = load_user_logs(user)
            if logs is not None:
                st.dataframe(logs)
                csv_log = logs.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Your Log", data=csv_log, file_name=f"{user}_logs.csv", mime="text/csv")
            else:
                st.info("No logs found for this user.")
