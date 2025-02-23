import nltk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary data
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    
    # Determine sentiment category
    if sentiment_score['compound'] >= 0.05:
        sentiment = "Positive 😊"
    elif sentiment_score['compound'] <= -0.05:
        sentiment = "Negative 😡"
    else:
        sentiment = "Neutral 😐"

    return sentiment, sentiment_score

# Streamlit UI
st.title("💬 Sentiment Analysis App")

st.write("Analyze the sentiment of your text with real-time visualizations.")

# User input
text = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if text.strip():
        sentiment, scores = analyze_sentiment(text)

        # Display sentiment result
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"**Sentiment Scores:** {scores}")

        # Create sentiment score bar chart
        st.subheader("📊 Sentiment Score Breakdown")
        fig, ax = plt.subplots()
        ax.bar(scores.keys(), scores.values(), color=['red', 'green', 'blue', 'gray'])
        ax.set_ylim([-1, 1])
        st.pyplot(fig)

        # Generate a word cloud
        st.subheader("☁️ Word Cloud of Input Text")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please enter some text.")

# Chatbot-style Sentiment Analysis
st.header("💬 Chatbot-Style Sentiment Analysis")
chat_input = st.text_input("Enter a sentence:")
if st.button("Check Sentiment"):
    if chat_input.strip():
        chat_sentiment, chat_scores = analyze_sentiment(chat_input)
        st.write(f"Sentiment: {chat_sentiment}")
        st.write(f"Scores: {chat_scores}")

# File Upload for Bulk Analysis
st.header("📂 Batch Sentiment Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file (must have a 'text' column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" in df.columns:
        df["Sentiment"], df["Scores"] = zip(*df["text"].apply(analyze_sentiment))
        st.write(df)

        # Save results
        df.to_csv("sentiment_results.csv", index=False)
        st.success("✅ Sentiment analysis completed! Results saved as 'sentiment_results.csv'.")

        # Sentiment Count Chart
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
        st.pyplot(fig)

    else:
        st.error("⚠️ CSV file must contain a 'text' column.")
