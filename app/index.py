
import streamlit as st
import joblib


model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

st.title('Message Mood Detector')


tweet = st.text_input("Enter message here to define mood:")

if tweet:
    processed = [' '.join(tweet.lower().split())]
    vectorized = vectorizer.transform(processed)
    prediction = model.predict(vectorized)[0]

    match prediction:
        case 0:
            pred_translated = 'negative'
        case 2:
            pred_translated = 'neutral'
        case 4:
            pred_translated = 'positive'
        case _:
            pred_translated = 'shit'

    probs = model.predict_proba(vectorized)[0]
    st.write(f'Mood: {pred_translated}, Confidence: {max(probs) * 100:.2f}%')
    





