# 📦 IMPORTS
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# NOTE: You MUST have run the previous notebook cell to generate
# `multilingual_app_reviews_cleaned.csv` for this code to work.

# 📥 Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
stop_words = set(stopwords.words('english'))

# ✅ 1. Load your pre-processed dataset
# This file contains the English-translated and cleaned reviews
try:
    df = pd.read_csv("multilingual_app_reviews_cleaned.csv")
    print("✅ Cleaned dataset loaded from file")
except FileNotFoundError:
    print("❌ ERROR: 'multilingual_app_reviews_cleaned.csv' not found.")
    print("Please run the translation and cleaning cell first to generate this file.")
    exit()

# FIX: Drop rows with NaN values to prevent errors with the models
df.dropna(subset=['review_text'], inplace=True)
print(f"✅ Dropped rows with NaN values. New size: {len(df)}")


# ✅ 2. BERTopic Topic Modeling
# Use the cleaned review text for better topic cohesion
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
topic_model = BERTopic(embedding_model=embedding_model)
topics, _ = topic_model.fit_transform(df['review_text'])
df['topic'] = topics
print("\n✅ BERTopic modeling complete.")


# ✅ 3. Sentiment Analysis using VADER
sid = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Assigns a sentiment label based on VADER's compound score."""
    if pd.isna(text):
        return 'neutral'
    score = sid.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['review_text'].apply(get_sentiment)
print("✅ VADER sentiment analysis complete.")


# ✅ 4. Load FLAN-T5 (or BART) for response generation
# NOTE: This model is large and will be slow.
# We will only generate responses for a small sample.
print("\n⏳ Loading FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
print("✅ FLAN-T5 model loaded.")


# ✅ 5. Define the response generation function
def generate_response_enhanced(topic, sentiment, review):
    """
    Generates a polite and empathetic customer support response
    using a pre-defined prompt template and a generative model.
    """
    # IMPROVED PROMPT
    prompt = f"""
You are a helpful and professional customer support agent for a mobile app.
A customer has left a {sentiment} review for a mobile app.
The review is about topic #{topic} and says: "{review}"

As a customer support agent, draft a polite and empathetic response to this review.
Make sure the response is human-like, unique, and not repetitive.
The response should be 1-2 sentences.
"""
    # ADJUSTED GENERATION PARAMETERS
    inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=80, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ✅ 6. Apply response generation to a small sample
# This is a critical step to avoid a very long runtime or crashes.
sample_size = 5
sample_df = df.sample(n=sample_size, random_state=42)

sample_df['auto_response'] = sample_df.apply(
    lambda row: generate_response_enhanced(row['topic'], row['sentiment'], row['review_text']),
    axis=1
)

# ✅ 7. Data Visualization
print("\n📈 Generating visualizations...")

# Pie chart: Sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
sentiment_counts.plot.pie(autopct='%1.1f%%', title='Sentiment Distribution')
plt.ylabel("")
plt.show()

# Word cloud for NEGATIVE reviews
negative_text = " ".join(df[df['sentiment'] == 'negative']['review_text'])
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.figure()
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis("off")
plt.title("Negative Review Word Cloud")
plt.show()

# Word cloud for POSITIVE reviews
positive_text = " ".join(df[df['sentiment'] == 'positive']['review_text'])
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure()
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis("off")
plt.title("Positive Review Word Cloud")
plt.show()

# ✅ 8. Show sample output with all information
print("\n🤖 Sample AI Support Responses:\n")
for i in range(sample_size):
    row = sample_df.iloc[i]
    print(f"🗣️ Sample Review : {row['review_text']}")
    print(f"📌 Topic ID    : {row['topic']}")
    print(f"❤️ Sentiment    : {row['sentiment']}")
    print(f"💬 Auto Response : {row['auto_response']}")
    print("-" * 80)

print("✅ Done!")

# ----------------------------------------------------------------------
# 🚀 Optional Streamlit App for Interactive Analysis
# To run this app, save the code to a file named 'app.py' and run:
# !streamlit run app.py
# Note: You'll need to install streamlit and googletrans first:
# !pip install streamlit googletrans==4.0.0-rc1

import streamlit as st
from googletrans import Translator

# Initialize translator, models, and NLTK resources
@st.cache_resource
def load_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    stop_words = set(stopwords.words('english'))
    translator = Translator()
    
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    topic_model = BERTopic(embedding_model=embedding_model)

    sid = SentimentIntensityAnalyzer()
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    return translator, topic_model, sid, tokenizer, model, stop_words

translator, topic_model, sid, tokenizer, model, stop_words = load_resources()

# Streamlit App UI
st.title("Customer Review Analyzer")
st.write("Input a customer review in any language and get an AI-generated response.")

user_review = st.text_area("Enter a review here:", height=150)

if st.button("Analyze Review"):
    if user_review:
        # Language Detection and Translation
        st.write("---")
        with st.spinner("Detecting language and translating..."):
            try:
                detected_lang = translator.detect(user_review).lang
                st.info(f"Detected Language: **{detected_lang}**")
                
                translated_review = translator.translate(user_review, dest='en').text
                st.markdown(f"**Translated Review:** {translated_review}")
            except Exception as e:
                st.error(f"Translation failed: {e}")
                translated_review = user_review

        # Preprocessing
        cleaned_review = translated_review.lower()
        cleaned_review = re.sub(r'[^a-zA-Z\s]', '', cleaned_review)
        words = cleaned_review.split()
        cleaned_review = " ".join([word for word in words if word not in stop_words])
        
        # Topic Modeling
        with st.spinner("Analyzing topic..."):
            topic_id, _ = topic_model.fit_transform([cleaned_review])
            st.success(f"**Detected Topic ID:** {topic_id[0]}")

        # Sentiment Analysis
        with st.spinner("Analyzing sentiment..."):
            sentiment_score = sid.polarity_scores(cleaned_review)['compound']
            if sentiment_score >= 0.05:
                sentiment = 'positive'
            elif sentiment_score <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            st.success(f"**Sentiment:** {sentiment}")

        # Generative Response
        with st.spinner("Generating response..."):
            prompt = f"""
            You are a helpful and professional customer support agent for a mobile app.
            A customer has left a {sentiment} review about topic #{topic_id[0]}.
            Here is what they said: "{translated_review}"

            As a customer support agent, draft a polite and empathetic response to this review.
            The response should be 1-2 sentences and not repetitive.
            """
            inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_new_tokens=80, num_beams=5, early_stopping=True)
            auto_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Auto-generated Response")
        response_text = st.text_area("Response", value=auto_response, height=100)

        # Copy to clipboard button
        st.markdown(f"""
            <button onclick="navigator.clipboard.writeText('{response_text}');">
                Copy Reply
            </button>
        """, unsafe_allow_html=True)
        
    else:
        st.warning("Please enter a review to analyze.")
