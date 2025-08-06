import streamlit as st
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

# Load models once
@st.cache_resource
def load_models():
    sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return sentiment_pipe, flan_tokenizer, flan_model, embedder

sentiment_pipe, flan_tokenizer, flan_model, embedder = load_models()

# Translate using LibreTranslate API
def translate_to_english(text):
    try:
        response = requests.post(
            "https://libretranslate.com/translate",
            data={
                "q": text,
                "source": "auto",
                "target": "en",
                "format": "text"
            },
            headers={
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )
        if response.status_code == 200:
            return response.json()["translatedText"]
        else:
            return text
    except Exception as e:
        print("Translation failed:", e)
        return text

# Smart response generator
def generate_response(text, sentiment, topic=None):
    prompt = (
        f"A customer left the following {sentiment.lower()} review: \"{text}\".\n"
        f"Write a professional and empathetic response as a customer service representative."
    )
    if topic:
        prompt += f" The review is about {topic}."
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = flan_model.generate(**inputs, max_new_tokens=80)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Classify star rating to sentiment
def classify_sentiment(score):
    return "POSITIVE" if score > 3 else "NEUTRAL" if score == 3 else "NEGATIVE"

# Cosine similarity-based clustering
def cluster_by_similarity(embeddings, threshold=0.85):
    clusters = []
    assigned = set()

    for i, emb in enumerate(embeddings):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, len(embeddings)):
            if j not in assigned:
                sim = cosine_similarity([emb], [embeddings[j]])[0][0]
                if sim >= threshold:
                    cluster.append(j)
                    assigned.add(j)
        clusters.append(cluster)

    labels = [0] * len(embeddings)
    for topic_id, group in enumerate(clusters):
        for index in group:
            labels[index] = topic_id
    return labels

# Streamlit UI
st.set_page_config(page_title="Review Analyzer", layout="wide")
st.title("🌍 Multilingual Review Response Generator")

review_input = st.text_area("✍️ Paste reviews (one per line)", height=200)

if st.button("Analyze"):
    raw_reviews = [r.strip() for r in review_input.split("\n") if r.strip()]
    if not raw_reviews:
        st.warning("Enter at least one review.")
        st.stop()

    st.markdown("---")
    translated_reviews, sentiments, topics, detected_langs = [], [], [], []

    for i, review in enumerate(raw_reviews):
        lang = detect(review)
        translated = translate_to_english(review) if lang != 'en' else review
        detected_langs.append(lang)

        sentiment_result = sentiment_pipe(translated)[0]
        score = int(sentiment_result['label'].split()[0])
        sentiment = classify_sentiment(score)

        sentiments.append(sentiment)
        translated_reviews.append(translated)

        st.markdown(f"**Review {i+1}:**")
        st.markdown(f"- 🌐 Language: `{lang}`")
        if lang != 'en':
            st.markdown(f"- 🗣️ Translated: `{translated}`")
        st.markdown(f"- 😊 Sentiment: `{sentiment}` ({score} stars)")

    # Clustering
    embeddings = embedder.encode(translated_reviews)
    cluster_labels = cluster_by_similarity(embeddings)

    for i, (review, sentiment, label) in enumerate(zip(translated_reviews, sentiments, cluster_labels)):
        topic = f"Topic {label + 1}"
        topics.append(topic)
        response = generate_response(review, sentiment, topic)

        st.markdown(f"**🧠 Topic:** `{topic}`")
        st.text_area("✍️ Suggested Response:", value=response, key=f"response_{i}", height=100)
        st.markdown("---")

    # Pie chart
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct="%1.1f%%")
    st.pyplot(fig)

    # Word clouds
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
