import streamlit as st
import joblib

model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

st.title('Tweet Mood Detector')

tweet = st.text_area('Input Twit:')

if tweet:
    processed = [' '.join(tweet.lower().split())]
    vectorized = vectorizer.transform(processed)
    prediction = model.predict(vectorized)[0]
    st.write(f"Mood: {prediction}")