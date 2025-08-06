import streamlit as st
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re

# Load models once
@st.cache_resource
def load_models():
    sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return sentiment_pipe, flan_tokenizer, flan_model, embedder

sentiment_pipe, flan_tokenizer, flan_model, embedder = load_models()

# Load translation models (for any lang to English)
@st.cache_resource
def load_translation_model():
    model_name = 'Helsinki-NLP/opus-mt-mul-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

trans_tokenizer, trans_model = load_translation_model()

# Translate using MarianMT
def translate_to_english(text):
    batch = trans_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    generated = trans_model.generate(**batch)
    return trans_tokenizer.decode(generated[0], skip_special_tokens=True)

# Smart response
def generate_response(text, sentiment, topic=None):
    prompt = (
        f"You're a customer support agent. A customer said: \"{text}\". "
        f"Generate a polite and empathetic response." +
        (f" The topic of the review is '{topic}'." if topic else "")
    )
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = flan_model.generate(**inputs, max_new_tokens=80)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sentiment from stars
def classify_sentiment(score):
    return "POSITIVE" if score > 3 else "NEUTRAL" if score == 3 else "NEGATIVE"

# Clustering
def cluster_reviews(reviews, n_clusters=3):
    embeddings = embedder.encode(reviews)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    return labels

# App UI
st.set_page_config(page_title="Review Analyzer", layout="wide")
st.title("🌍 Multilingual Review Response Generator")

review_input = st.text_area("✍️ Paste reviews (one per line)", height=200)

if st.button("Analyze"):
    raw_reviews = [r.strip() for r in review_input.split("\n") if r.strip()]
    if not raw_reviews:
        st.warning("Enter at least one review.")
        st.stop()

    st.markdown("---")
    translated_reviews, sentiments, topics = [], [], []

    # 1. Translate & Sentiment
    for i, review in enumerate(raw_reviews):
        lang = detect(review)
        translated = translate_to_english(review) if lang != 'en' else review
        sentiment_result = sentiment_pipe(translated)[0]
        score = int(sentiment_result['label'].split()[0])
        sentiment = classify_sentiment(score)

        sentiments.append(sentiment)
        translated_reviews.append(translated)

        st.markdown(f"**Review {i+1}:**")
        st.markdown(f"- 🌐 Language: `{lang}`")
        if lang != 'en':
            st.markdown(f"- 🗣️ Translated: {translated}")
        st.markdown(f"- 😊 Sentiment: `{sentiment}` ({score} stars)")

    # 2. Clustering
    cluster_labels = cluster_reviews(translated_reviews, n_clusters=3)
    for i, (review, sentiment, label) in enumerate(zip(translated_reviews, sentiments, cluster_labels)):
        topic = f"Topic {label + 1}"
        topics.append(topic)
        response = generate_response(review, sentiment, topic)

        st.markdown(f"**🧠 Topic:** `{topic}`")
        st.text_area("✍️ Suggested Response:", value=response, key=f"response_{i}", height=100)
        st.markdown("---")

    # 3. Visualization
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct="%1.1f%%")
    st.pyplot(fig)

    st.subheader("☁️ Word Cloud per Topic")
    for label in set(cluster_labels):
        topic_reviews = [r for i, r in enumerate(translated_reviews) if cluster_labels[i] == label]
        text = " ".join(topic_reviews)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.markdown(f"**🧠 Topic {label + 1}**")
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
