import streamlit as st
from langdetect import detect
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Translation using LibreTranslate API
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

# Load models once
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model)
    return sentiment_model, topic_model

sentiment_model, topic_model = load_models()

# App title
st.title("🌍 Multilingual Review Response Generator")

# Text input
review_input = st.text_area("✍️ Enter review(s) - one per line:", height=200)

if st.button("Analyze"):
    reviews = [r.strip() for r in review_input.strip().split("\n") if r.strip()]
    sentiments = []
    topics_list = []

    if len(reviews) == 1:
        # Add dummy review to prevent UMAP error
        reviews.append("This is a dummy review to enable topic modeling.")

    st.markdown("---")
    for review in reviews:
        if review == "This is a dummy review to enable topic modeling.":
            continue

        # Detect language
        lang = detect(review)
        st.markdown(f"**🌐 Detected Language:** {lang}")

        # Translate if not English
        translated_review = translate_to_english(review, lang) if lang != "en" else review
        if lang != "en":
            st.markdown(f"**🗣️ Translated Review:** {translated_review}")

        # Sentiment
        result = sentiment_model(translated_review)[0]
        sentiment = result['label']
        sentiments.append(sentiment)
        st.markdown(f"**😊 Sentiment:** {sentiment}")

        # Collect for topic modeling
        topics_list.append(translated_review)

        # Response
        response = f"As a support agent: We appreciate your feedback. Regarding your comment: \"{translated_review}\", we will work on it. Thank you!"
        editable = st.text_area("✍️ Suggested Response (editable):", value=response, height=100)

        st.markdown("---")

    # Topic Modeling
    topics, _ = topic_model.fit_transform(topics_list)
    for i, topic in enumerate(topics):
        topic_info = topic_model.get_topic(topic)
        topic_label = topic_info[0][0] if topic_info else "N/A"
        st.markdown(f"**🧠 Topic for Review {i+1}:** {topic_label}")

    # Pie Chart for Sentiment Distribution
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct="%1.1f%%", startangle=90)
    st.pyplot(fig)

    # WordCloud for all reviews
    st.subheader("☁️ WordCloud of Translated Reviews")
    all_text = " ".join(topics_list)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)
