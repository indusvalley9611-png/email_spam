import streamlit as st
import joblib

# LOAD MODEL + VECTORIZER
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Spam Detector", layout="wide")

st.title("📩 Email Spam Detection App")

st.write("Enter a message to check if it's spam or not.")

# INPUT
message = st.text_area("Enter your message here")

# BUTTON
if st.button("Check Spam", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        vec = vectorizer.transform([message])
        prediction = model.predict(vec)[0]

        if prediction.lower() == "spam":
            st.error("🚫 This is SPAM")
        else:
            st.success("✅ This is NOT Spam")
