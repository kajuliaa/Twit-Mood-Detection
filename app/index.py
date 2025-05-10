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
    print(prediction)
    pred_translated = ''
    match prediction:
        case 0:
            pred_translated = 'negative'
        case 2:
            pred_translated = 'neutral'
        case 4:
            pred_translated = 'positive'
        case _:
            pred_translated = 'shit'
    st.write(f"Mood: {pred_translated}")