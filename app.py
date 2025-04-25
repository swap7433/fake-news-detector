import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('Model/model.pkl')
vectorizer = joblib.load('Model/vectorizer.pkl')

# Function for Fake News Detection
def classify_fake_news(text):
    # Use the same vectorizer that was used during training
    text_vectorized = vectorizer.transform([text])  # Vectorize the input text
    
    # Predict using the trained model
    prediction = model.predict(text_vectorized)
    
    # Return the result based on prediction
    if prediction == 1:
        return "Real News", "green"
    else:
        return "Fake News", "red"

# Streamlit UI elements
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# Title and subtitle
st.title("Fake News Detection")
st.markdown("### Identify whether a news article is real or fake.")

# Custom CSS for smaller text area
st.markdown("""
    <style>
        .textarea {
            width: 500px;
            height: 150px;
        }
    </style>
""", unsafe_allow_html=True)

# User input section for the news article
input_text = st.text_area("Enter the news article text here:", height=150, key="news_article", help="Please enter the news article you want to verify.", max_chars=1000)

# Add a Detect button
if st.button("Detect"):
    if input_text:
        # Fake News Detection
        result, color = classify_fake_news(input_text)
        st.subheader("Fake News Detection Result:")
        st.markdown(f"<p style='color:{color}; font-size: 18px;'><strong>{result}</strong></p>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a news article to get started.")
