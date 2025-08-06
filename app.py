import streamlit as st
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch

# ---------------------- Translation Function ---------------------- #
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

# ---------------------- Load Models ---------------------- #
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return sentiment_model, topic_model, tokenizer, generator_model

sentiment_model, topic_model, tokenizer, generator_model = load_models()

# ---------------------- Generate Auto Response ---------------------- #
def generate_response(review_text):
    prompt = f"You are a polite customer support agent. A customer said: '{review_text}'. Write a helpful and empathetic response."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = generator_model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------- Streamlit UI ---------------------- #
st.title("🌍 Multilingual Review Response Generator")
review_input = st.text_area("✍️ Enter review(s) - one per line:", height=200)

if st.button("Analyze"):
    reviews = [r.strip() for r in review_input.strip().split("\n") if r.strip()]
    sentiments = []
    topics_list = []

    if len(reviews) == 1:
        reviews.append("This is a dummy review to enable topic modeling.")

    st.markdown("---")
    for review in reviews:
        if review == "This is a dummy review to enable topic modeling.":
            continue

        # 🌐 Language Detection
        lang = detect(review)
        st.markdown(f"**🌐 Detected Language:** {lang}")

        # 🌍 Translation
        translated_review = translate_to_english(review, lang) if lang != "en" else review
        if lang != "en":
            st.markdown(f"**🗣️ Translated Review:** {translated_review}")

        # 😊 Sentiment
        result = sentiment_model(translated_review)[0]
        sentiment = result['label']
        sentiments.append(sentiment)
        st.markdown(f"**😊 Sentiment:** {sentiment}")

        # Collect review for topic modeling
        topics_list.append(translated_review)

        # 🤖 Auto-generated Response
        response = generate_response(translated_review)
        editable_response = st.text_area("✍️ Suggested Response (editable):", value=response, height=100)

        st.markdown("---")

    # 🧠 Topic Modeling
    topics, _ = topic_model.fit_transform(topics_list)
    for i, topic in enumerate(topics):
        topic_info = topic_model.get_topic(topic)
        topic_label = topic_info[0][0] if topic_info else "N/A"
        st.markdown(f"**🧠 Topic for Review {i+1}:** {topic_label}")

    # 📊 Pie Chart: Sentiment Distribution
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct="%1.1f%%", startangle=90)
    st.pyplot(fig)

    # ☁️ Word Cloud
    st.subheader("☁️ Word Cloud of Translated Reviews")
    all_text = " ".join(topics_list)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)
