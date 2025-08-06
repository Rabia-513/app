import streamlit as st
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 🔁 Translate to English (Improved with fallback and logging)
def translate_to_english(text, source_lang):
    try:
        response = requests.post(
            "https://libretranslate.com/translate",
            json={
                "q": text,
                "source": source_lang,
                "target": "en",
                "format": "text"
            },
            timeout=10
        )
        result = response.json()
        if "translatedText" in result and result["translatedText"].strip().lower() != text.strip().lower():
            return result["translatedText"]
        else:
            return text
    except Exception as e:
        print("Translation failed:", e)
        return text

# 🔁 Load models
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    response_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return sentiment_model, topic_model, tokenizer, response_model

sentiment_model, topic_model, tokenizer, response_model = load_models()

# 🤖 Generate LLM-based response
def generate_response(review_text):
    prompt = f"You are a helpful and polite customer support agent. A customer said: \"{review_text}\". Write a kind, empathetic response."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = response_model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🚀 UI
st.set_page_config(page_title="Multilingual Review Analyzer", layout="wide")
st.title("🌍 Multilingual Review Response Generator")
review_input = st.text_area("✍️ Enter review(s) - one per line:", height=200)

if st.button("Analyze"):
    raw_reviews = [r.strip() for r in review_input.strip().split("\n") if r.strip()]

    if not raw_reviews:
        st.warning("Please enter at least one review.")
    else:
        st.markdown("---")
        translated_reviews = []
        sentiments = []
        displayed_reviews = []

        # Add dummy if only one review (for BERTopic)
        modeling_reviews = raw_reviews.copy()
        if len(raw_reviews) == 1:
            modeling_reviews.append("dummy review to enable topic modeling")

        for review in raw_reviews:
            # 🌐 Language detection
            lang = detect(review)
            st.markdown(f"**🌐 Detected Language:** {lang}")

            # 🌍 Translate
            translated = translate_to_english(review, lang) if lang != "en" else review
            if lang != "en":
                st.markdown(f"**🗣️ Translated Review:** {translated}")

            if lang != "en" and translated.strip().lower() == review.strip().lower():
                st.warning("⚠️ Translation API might have failed. Review not translated.")

            # 😊 Sentiment
            result = sentiment_model(translated)[0]
            sentiment = result['label']
            sentiments.append(sentiment)
            st.markdown(f"**😊 Sentiment:** {sentiment}")

            # 🤖 LLM-generated response
            response = generate_response(translated)
            st.text_area("✍️ Suggested Response (editable):", value=response, height=100)

            translated_reviews.append(translated)
            displayed_reviews.append(review)

            st.markdown("---")

        # 🧠 Topic Modeling
        st.subheader("🧠 Topics")
     # Filter out empty or dummy-only reviews
valid_reviews = [r for r in translated_reviews if r.strip() and "dummy" not in r.lower()]
if len(valid_reviews) >= 2:
    topics, _ = topic_model.fit_transform(valid_reviews)
    st.subheader("🧠 Topics")
    for i, topic in enumerate(topics[:len(displayed_reviews)]):
        topic_words = topic_model.get_topic(topic)
        topic_label = topic_words[0][0] if topic_words else "N/A"
        st.markdown(f"**Topic for Review {i+1}:** {topic_label}")
else:
    st.warning("⚠️ Not enough valid reviews to perform topic modeling. Please enter at least 2 reviews.")


        # 📊 Sentiment Distribution Pie Chart
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct="%1.1f%%", startangle=90)
        st.pyplot(fig)

        # ☁️ Word Cloud
        st.subheader("☁️ Word Cloud of Translated Reviews")
        all_text = " ".join(translated_reviews)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

