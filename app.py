import streamlit as st
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 🔁 Translate to English (fallback-safe)
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
        return result.get("translatedText", text)
    except Exception as e:
        print("Translation error:", e)
        return text

# ✅ Load models
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    response_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return sentiment_model, tokenizer, response_model

sentiment_model, tokenizer, response_model = load_models()

# 🤖 LLM-based response (based on sentiment)
def generate_response(text, sentiment):
    if sentiment == "POSITIVE":
        prompt = f"You are a helpful support agent. The customer said: \"{text}\". Reply with appreciation and positivity."
    else:
        prompt = f"You are a polite support agent. The customer had a bad experience: \"{text}\". Respond with empathy and offer to improve."

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = response_model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🖼️ UI Setup
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

        for i, review in enumerate(raw_reviews):
            # 🌐 Language detection
            lang = detect(review)
            st.markdown(f"**🌐 Detected Language:** {lang}")

            # 🌍 Translate
            translated = translate_to_english(review, lang) if lang != "en" else review
            if lang != "en":
                st.markdown(f"**🗣️ Translated Review:** {translated}")

            # 😊 Sentiment Analysis
            result = sentiment_model(translated)[0]
            sentiment = result['label']
            sentiments.append(sentiment)
            st.markdown(f"**😊 Sentiment:** {sentiment}")

            # 🤖 Response based on sentiment
            response = generate_response(translated, sentiment)
            st.text_area("✍️ Suggested Response (editable):", value=response, height=100, key=f"response_{i}")

            translated_reviews.append(translated)
            st.markdown("---")

        # 📊 Sentiment Pie Chart
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
