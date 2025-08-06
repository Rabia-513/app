import streamlit as st
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# ✅ Check if text is meaningful
def is_text_meaningful(text):
    return bool(re.search(r'[a-zA-Z]', text)) and len(text.split()) >= 3

# 🔁 Translate to English using LibreTranslate API
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
        translated = result.get("translatedText", "").strip()
        if not translated:
            st.warning("⚠️ Translation failed or empty.")
        elif translated.lower() == text.strip().lower():
            st.info("ℹ️ Review may already be in English or not translated meaningfully.")
        return translated if translated else text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# ✅ Load models with caching
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    response_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return sentiment_model, tokenizer, response_model

sentiment_model, tokenizer, response_model = load_models()

# 🤖 Generate smart response
def generate_response(text):
    prompt = (
        f"You're a customer support agent. A customer said: \"{text}\". "
        "Generate a polite and empathetic response."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = response_model.generate(**inputs, max_new_tokens=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🚀 UI Setup
st.set_page_config(page_title="Multilingual Review Response Generator", layout="wide")
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
            lang = detect(review)
            st.markdown(f"**🌐 Detected Language:** `{lang}`")

            # Translate if needed
            translated = translate_to_english(review, lang) if lang != "en" else review
            
            if lang != "en":
                st.markdown(f"**🗣️ Translated Review:** {translated}")

            # 😊 Sentiment Detection (1–5 stars)
            result = sentiment_model(translated)[0]
            score = int(result['label'].split()[0])
            sentiment = "POSITIVE" if score > 3 else "NEGATIVE"
            sentiments.append(sentiment)
            st.markdown(f"**😊 Sentiment:** {sentiment} ({score} stars)")

            # 🤖 Suggested Response
            response = generate_response(translated)
            st.text_area("✍️ Suggested Response (editable):", value=response, height=100, key=f"response_{i}")

            translated_reviews.append(translated)
            st.markdown("---")

        # 📊 Pie Chart
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
