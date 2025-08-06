import streamlit as st
from langdetect import detect
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, M2M100Tokenizer
)
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models
@st.cache_resource
def load_models():
    sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return sentiment_pipe, flan_tokenizer, flan_model, embedder

sentiment_pipe, flan_tokenizer, flan_model, embedder = load_models()

# Load M2M100 translation model (no sentencepiece needed)
@st.cache_resource
# Load translation model
@st.cache_resource
def load_translation_model():
    model_name = 'Helsinki-NLP/opus-mt-mul-en'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Translate using MarianMT
def translate_to_english(text):
    inputs = trans_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    outputs = trans_model.generate(**inputs)
    return trans_tokenizer.decode(outputs[0], skip_special_tokens=True)


# Generate AI response
def generate_response(review, sentiment, topic):
    prompt = (
        f"A customer wrote a {sentiment.lower()} review: '{review}'. "
        f"The topic is '{topic}'. "
        "Write a polite, helpful customer service response to this review."
    )
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True)
    output = flan_model.generate(**inputs, max_new_tokens=80)
    return flan_tokenizer.decode(output[0], skip_special_tokens=True)

# Classify sentiment
def classify_sentiment(score):
    return "POSITIVE" if score > 3 else "NEUTRAL" if score == 3 else "NEGATIVE"

# Cosine similarity clustering
def cluster_by_similarity(embeddings, threshold=0.85):
    clusters = []
    assigned = set()
    for i, emb in enumerate(embeddings):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, len(embeddings)):
            if j not in assigned and cosine_similarity([emb], [embeddings[j]])[0][0] >= threshold:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)

    labels = [0] * len(embeddings)
    for topic_id, group in enumerate(clusters):
        for idx in group:
            labels[idx] = topic_id
    return labels

# Streamlit UI
st.set_page_config(page_title="Multilingual Review Analyzer", layout="wide")
st.title("🌍 Multilingual Review Response Generator")

review_input = st.text_area("✍️ Paste reviews (one per line)", height=200)

if st.button("Analyze"):
    raw_reviews = [r.strip() for r in review_input.split("\n") if r.strip()]
    if not raw_reviews:
        st.warning("Enter at least one review.")
        st.stop()

    st.markdown("---")
    translated_reviews, sentiments, topics, languages = [], [], [], []

    # Step 1: Translate + Sentiment
    for i, review in enumerate(raw_reviews):
        lang = detect(review)
        languages.append(lang)

       translated = translate_to_english(review) if lang != "en" else review

        sentiment_result = sentiment_pipe(translated)[0]
        score = int(sentiment_result["label"].split()[0])
        sentiment = classify_sentiment(score)

        sentiments.append(sentiment)
        translated_reviews.append(translated)

        st.markdown(f"**Review {i+1}:**")
        st.markdown(f"- 🌐 Language: `{lang}`")
        if lang != "en":
            st.markdown(f"- 🗣️ Translated: `{translated}`")
        st.markdown(f"- 😊 Sentiment: `{sentiment}` ({score} stars)")

    # Step 2: Clustering
    embeddings = embedder.encode(translated_reviews)
    cluster_labels = cluster_by_similarity(embeddings, threshold=0.85)

    for i, (review, sentiment, label) in enumerate(zip(translated_reviews, sentiments, cluster_labels)):
        topic = f"Topic {label + 1}"
        topics.append(topic)
        response = generate_response(review, sentiment, topic)

        st.markdown(f"**🧠 Topic:** `{topic}`")
        st.text_area("✍️ Suggested Response:", value=response, key=f"response_{i}", height=100)
        st.markdown("---")

    # Step 3: Charts
    st.subheader("📊 Sentiment Distribution")
    counts = {s: sentiments.count(s) for s in set(sentiments)}
    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
    st.pyplot(fig)

    st.subheader("☁️ Word Cloud per Topic")
    for label in set(cluster_labels):
        topic_text = " ".join([r for i, r in enumerate(translated_reviews) if cluster_labels[i] == label])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(topic_text)
        st.markdown(f"**🧠 Topic {label + 1}**")
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)


