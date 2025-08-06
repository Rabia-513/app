import streamlit as st
from langdetect import detect
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import requests

# Lightweight translation using LibreTranslate API
def translate_to_english(text, source_lang):
    try:
        response = requests.post("https://libretranslate.com/translate", json={
            "q": text,
            "source": source_lang,
            "target": "en",
            "format": "text"
        })
        return response.json()["translatedText"]
    except:
        return text

# Load models (cache for performance)
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # smaller
    topic_model = BERTopic(embedding_model=embedding_model)
    return sentiment_model, topic_model

sentiment_model, topic_model = load_models()

# Title
st.title("🌍 Multilingual Review Response Generator")

# Input
review_input = st.text_area("✍️ Enter review(s):", height=200)

if st.button("Analyze"):
    reviews = review_input.strip().split('\n')
    for review in reviews:
        if not review.strip():
            continue

        # Language detection
        lang = detect(review)
        st.markdown(f"**🌐 Detected Language:** {lang}")

        # Translate if not English
        translated_review = review
        if lang != 'en':
            translated_review = translate_to_english(review, lang)
            st.markdown(f"**🗣️ Translated Review:** {translated_review}")

        # Sentiment
        sentiment_result = sentiment_model(translated_review)[0]
        sentiment = sentiment_result['label']
        st.markdown(f"**😊 Sentiment:** {sentiment}")

        # Topic Modeling
      # Topic Modeling (UMAP requires at least 2 samples)
text_for_topic_modeling = [translated_review]
if len(text_for_topic_modeling) == 1:
    text_for_topic_modeling.append("This is a dummy review to enable topic modeling.")

topics, _ = topic_model.fit_transform(text_for_topic_modeling)
topic_label = topic_model.get_topic(topics[0])[0][0] if topics else "N/A"
st.markdown(f"**🧠 Topic:** {topic_label}")

        topic_label = topic_model.get_topic(topics[0])[0][0] if topics else "N/A"
        st.markdown(f"**🧠 Topic:** {topic_label}")

        # Auto Response
        response = f"As a support agent: We appreciate your feedback. Regarding your comment: \"{translated_review}\", we will work on it. Thank you!"
        st.text_area("✍️ Suggested Response:", value=response, height=100)

if __name__ == "__main__":
    st.write("App loaded successfully")

