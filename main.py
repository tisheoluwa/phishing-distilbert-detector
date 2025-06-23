import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load tokenizer and model from saved directory
MODEL_DIR = "phishing_bert_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def predict(text):
    """
    Predict whether the given text is phishing or not.
    Returns:
        label (int): 0 for legitimate, 1 for phishing
        confidence (float): confidence score between 0 and 1
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    return predicted_class.item(), round(confidence.item(), 4)


# Example usage (for testing)
if __name__ == "__main__":
    sample = "Click here to verify your account immediately or it will be suspended."
    label, confidence = predict(sample)
    print("Prediction:", "Phishing" if label == 1 else "Legitimate")
    print("Confidence:", confidence)
