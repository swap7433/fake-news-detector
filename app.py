# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:59:50 2025

@author: sonaw
"""
import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Set up the Streamlit page
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection App")
st.write("Enter a news article and click predict to find out if it is **Real** or **Fake**!")

# Input text from the user
input_text = st.text_area("News Text", height=300)

# Function to summarize the text (optional)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 2)  # Summarize to 2 sentences
    return " ".join(str(sentence) for sentence in summary)

# Handle form submissions
if st.button("Predict"):
    # Vectorize the input text using the loaded vectorizer
    transformed = vectorizer.transform([input_text])

    # Predict using the loaded model
    prediction = model.predict(transformed)
    
    # Display the result: "Real News" or "Fake News"
    result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"
    st.subheader("Prediction:")
    st.success(result)

    # Show the summary of the article (optional feature)
    if st.checkbox("Show Article Summary"):
        st.subheader("Summary:")
        st.info(summarize_text(input_text))

    # Save prediction history in session state (to persist across interactions)
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append((input_text, result))

# Show prediction history
if st.session_state.history:
    st.subheader("🕒 Prediction History")
    for i, (text, res) in enumerate(st.session_state.history[::-1], 1):
        st.markdown(f"**{i}.** `{res}` – {text[:75]}...")

