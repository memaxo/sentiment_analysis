# Sentiment Analysis Project

A sentiment analysis tool that predicts the sentiment (positive, negative, or neutral) of text input. This project includes both a Flask web application and a Streamlit application.

## Files

- `sentiment_analysis.py`: Core module containing the preprocessing and sentiment prediction logic
- `sentiment_webapp.py`: Flask web application for sentiment analysis
- `streamlit_app.py`: Streamlit web application for sentiment analysis
- `requirements.txt`: Dependencies for the project
- `render.yaml`: Configuration for deployment on Render

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```
   
   Or run the Flask app:
   ```
   python sentiment_webapp.py
   ```

## Model

The project uses a logistic regression model trained on Twitter sentiment data. 