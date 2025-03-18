#!/usr/bin/env python3
# Sentiment Analysis Streamlit App

import os
import sys
import pickle
import streamlit as st
import numpy as np
import nltk
import importlib.util

# Ensure correct path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Download required NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Dynamically import the sentiment analysis module
spec = importlib.util.spec_from_file_location(
    "sentiment_analysis", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_analysis.py")
)
sentiment_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sentiment_analysis)

# Make required functions available
predict_sentiment = sentiment_analysis.predict_sentiment
preprocess_text = sentiment_analysis.preprocess_text
tokenizer = sentiment_analysis.tokenizer
tokenizer_porter = sentiment_analysis.tokenizer_porter

# Model path
MODEL_PATH = 'sentiment_model.pkl'

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: {MODEL_PATH} not found. Please train the model first by running sentiment_analysis.py")

# App title and description
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="centered"
)

# Main app UI
st.title("Sentiment Analysis Tool")
st.write("Enter text below to analyze its sentiment (positive, negative, or neutral)")

# Text input
text_input = st.text_area("", height=150, placeholder="Type or paste text here...")

# Options
with st.expander("Advanced Options"):
    show_score = st.checkbox("Show raw sentiment score", value=False)
    confidence_threshold = st.slider(
        "Neutral sentiment threshold", 
        min_value=0.1, 
        max_value=0.4, 
        value=0.2,
        help="Lower values mean fewer neutral predictions"
    )

# Analysis button
if st.button("Analyze Sentiment", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze")
    else:
        # Display spinner during prediction
        with st.spinner("Analyzing sentiment..."):
            try:
                # Get sentiment prediction using proper neutral threshold
                sentiment_score = predict_sentiment([text_input], MODEL_PATH, return_proba=True)[0]
                prediction = predict_sentiment([text_input], MODEL_PATH, return_proba=False, 
                                             neutral_threshold=confidence_threshold)[0]
                
                # Determine sentiment category from prediction
                if prediction == 1:  # Positive
                    sentiment = "Positive"
                    confidence = sentiment_score
                    color = "rgba(39, 174, 96, 0.2)"  # Green
                    emoji = "😃"
                elif prediction == 0:  # Negative
                    sentiment = "Negative"
                    confidence = 1 - sentiment_score
                    color = "rgba(231, 76, 60, 0.2)"  # Red
                    emoji = "😔"
                else:  # prediction == 2 (Neutral)
                    sentiment = "Neutral"
                    confidence = 1 - abs(sentiment_score - 0.5) * 2  # Scale confidence
                    color = "rgba(127, 140, 141, 0.2)"  # Gray
                    emoji = "😐"
                
                # Display result
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {color}; margin-bottom: 20px;">
                    <h2 style="text-align: center; margin: 0;">{emoji} {sentiment}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence
                st.write(f"**Confidence:** {confidence:.2f}")
                st.progress(float(confidence))
                
                # Show raw score if requested
                if show_score:
                    st.write(f"**Raw sentiment score:** {sentiment_score:.4f} (0 = Negative, 1 = Positive)")
                    
                # Show preprocessing result
                with st.expander("View text preprocessing"):
                    st.code(preprocess_text(text_input))
                    
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
                st.info("Make sure you've trained the model by running sentiment_analysis.py first")

# Example texts to try
with st.expander("Example texts to try"):
    examples = [
        "I absolutely love this product! It's amazing and works exactly as described.",
        "This is the worst experience I've ever had. Terrible customer service.",
        "The weather seems fine today. Not too hot, not too cold.",
        "I'm not sure if I liked it or not. It had some good parts but also some issues.",
        "Just tried the new iPhone and it's amazing! #technology #apple",
        "Customer service was terrible and the product broke after two days :("
    ]
    
    for example in examples:
        if st.button(example[:50] + "..." if len(example) > 50 else example):
            st.session_state.text_input = example
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit using logistic regression model. "
    "This app handles emojis, emoticons, hashtags, and Twitter-specific language."
) 