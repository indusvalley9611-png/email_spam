import streamlit as st
import joblib

# LOAD
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📩 Spam Detector")

msg = st.text_area("Enter message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Enter message")
    else:
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]

        if pred == "spam":
            st.error("🚫 Spam")
        else:
            st.success("✅ Not Spam")
