import streamlit as st
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# ✅ Helper: Check for meaningful translation
def is_text_meaningful(text):
    return bool(re.search(r'[a-zA-Z]', text)) and len(text.split()) >= 3

# ✅ Translation using LibreTranslate API
def translate_to_english(text, source_lang):
    try:
        response = requests.post(
            "https://libretranslate.de/translate",  # Stable mirror
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
        if not translated or translated.lower() == text.lower():
            return text, False  # fallback
        return translated, True
    except Exception as e:
        print("Translation error:", e)
        return text, False

# ✅ Load models (with better multilingual support and LLM)
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    response_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return sentiment_model, tokenizer, response_model

sentiment_model, tokenizer, response_model = load_models()

# ✅ Generate polite, helpful 2-line support response
def generate_response(text, sentiment):
    if sentiment == "POSITIVE":
        prompt = f"A customer said: \"{text}\". As a helpful support agent, write a short 2-line polite and appreciative response."
    else:
        prompt = f"A customer said: \"{text}\". As a support agent, write a short 2-line empathetic response acknowledging the issue and promising improvement."

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = response_model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Streamlit UI
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
            lang = detect(review)
            st.markdown(f"**🌐 Detected Language:** `{lang}`")

            translated, translated_ok = translate_to_english(review, lang) if lang != "en" else (review, False)
            if lang != "en":
                if not translated_ok:
                    st.warning("⚠️ Translation may have failed or was not required.")
                else:
                    st.markdown(f"**🗣️ Translated Review:** {translated}")

            # ✅ Sentiment detection using star-based scoring
            result = sentiment_model(translated)[0]
            score = int(result['label'][0])  # e.g., '4 stars' -> 4
            sentiment = "POSITIVE" if score > 3 else "NEGATIVE"
            sentiments.append(sentiment)
            st.markdown(f"**😊 Sentiment:** {sentiment} ({score} stars)")

            # ✅ Generate 2-line AI Response
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

        # ☁️ Word Cloud from Reviews
        st.subheader("☁️ Word Cloud of Translated Reviews")
        all_text = " ".join(translated_reviews)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
