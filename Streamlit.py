import streamlit as st
import torch
import pickle

# Load model and tokenizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Define label mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Define prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return label_map[prediction]

# Streamlit app
st.title("Stock Market Sentiment Analysis")

user_input = st.text_area("Enter a forum post or comment:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
