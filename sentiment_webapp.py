#!/usr/bin/env python3
# Sentiment Analysis Web Application

import os
import pickle
import re
from flask import Flask, render_template, request, jsonify
import nltk
import sys
import importlib.util

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Add the parent directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Dynamically import the entire sentiment_analysis module
spec = importlib.util.spec_from_file_location("sentiment_analysis", 
                                              os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                          "sentiment_analysis.py"))
sentiment_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sentiment_analysis)

# Make the tokenizer functions available in the global namespace
tokenizer = sentiment_analysis.tokenizer
tokenizer_porter = sentiment_analysis.tokenizer_porter
predict_sentiment = sentiment_analysis.predict_sentiment
preprocess_text = sentiment_analysis.preprocess_text

app = Flask(__name__)

# Check if model exists, otherwise inform user to train it first
MODEL_PATH = 'sentiment_model.pkl'
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Please train the model first by running sentiment_analysis.py")
    print("Continuing anyway for development purposes, but predictions won't work.")

@app.route('/')
def home():
    """Render the home page with the sentiment analysis form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process the user input and return sentiment prediction"""
    # Get text from request
    text = request.form.get('text', '')
    
    if not text.strip():
        return jsonify({
            'error': 'Please enter some text to analyze'
        })
    
    try:
        # Get neutral threshold from request or use default
        neutral_threshold = float(request.form.get('threshold', 0.2))
        
        # Get prediction directly using the updated predict_sentiment function
        sentiment_score = predict_sentiment([text], MODEL_PATH, return_proba=True)[0]
        prediction = predict_sentiment([text], MODEL_PATH, return_proba=False, 
                                     neutral_threshold=neutral_threshold)[0]
        
        # Map numeric prediction to sentiment label
        if prediction == 1:
            sentiment = "Positive"
            confidence = sentiment_score
        elif prediction == 0:
            sentiment = "Negative"
            confidence = 1 - sentiment_score
        else:  # prediction == 2
            sentiment = "Neutral"
            confidence = 1 - abs(sentiment_score - 0.5) * 2  # Scale confidence

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'confidence': float(f"{confidence:.2f}"),
            'score': float(f"{sentiment_score:.2f}")
        })
    
    except Exception as e:
        return jsonify({
            'error': f"Error processing text: {str(e)}"
        })

@app.route('/templates/index.html')
def serve_template():
    """Serve the template directly for development purposes"""
    return render_template('index.html')

# Create templates directory and index.html if they don't exist
def create_templates():
    """Create necessary templates and static files"""
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-height: 100px;
            margin-bottom: 15px;
            font-family: inherit;
            font-size: 16px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .positive {
            background-color: #d4f1e4;
            border-left: 4px solid #27ae60;
        }
        .negative {
            background-color: #fde4e4;
            border-left: 4px solid #e74c3c;
        }
        .neutral {
            background-color: #e9f2f9;
            border-left: 4px solid #7f8c8d;
        }
        .confidence {
            font-weight: bold;
            margin-top: 10px;
        }
        #loader {
            display: none;
            text-align: center;
            margin: 15px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background-color: #fde4e4;
            border-left: 4px solid #e74c3c;
            padding: 10px;
            margin-top: 15px;
            border-radius: 4px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis Tool</h1>
        <form id="sentimentForm">
            <textarea id="textInput" placeholder="Enter text to analyze sentiment..."></textarea>
            
            <div style="margin-bottom: 15px;">
                <label for="thresholdSlider" style="display: block; margin-bottom: 5px;">Neutral Threshold: <span id="thresholdValue">0.2</span></label>
                <input type="range" id="thresholdSlider" name="threshold" min="0.1" max="0.4" step="0.05" value="0.2" 
                       style="width: 100%;" oninput="document.getElementById('thresholdValue').textContent = this.value">
                <small style="color: #666; display: block; margin-top: 5px;">Higher values will classify more text as neutral.</small>
            </div>
            
            <button type="submit">Analyze Sentiment</button>
        </form>
        
        <div id="loader">
            <div class="spinner"></div>
        </div>
        
        <div id="error" class="error"></div>
        
        <div id="result">
            <h3>Sentiment: <span id="sentiment"></span></h3>
            <div class="confidence">Confidence: <span id="confidence"></span>%</div>
        </div>
    </div>

    <script>
        document.getElementById('sentimentForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const textInput = document.getElementById('textInput').value.trim();
            const resultDiv = document.getElementById('result');
            const loader = document.getElementById('loader');
            const errorDiv = document.getElementById('error');
            
            // Hide previous results and errors
            resultDiv.style.display = 'none';
            errorDiv.style.display = 'none';
            
            if (!textInput) {
                errorDiv.textContent = 'Please enter some text to analyze';
                errorDiv.style.display = 'block';
                return;
            }
            
            // Show loader
            loader.style.display = 'block';
            
            try {
                const formData = new FormData();
                formData.append('text', textInput);
                formData.append('threshold', document.getElementById('thresholdSlider').value);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Hide loader
                loader.style.display = 'none';
                
                if (data.error) {
                    errorDiv.textContent = data.error;
                    errorDiv.style.display = 'block';
                } else {
                    // Update result
                    document.getElementById('sentiment').textContent = data.sentiment;
                    document.getElementById('confidence').textContent = Math.round(data.confidence * 100);
                    
                    // Apply appropriate styling based on sentiment
                    resultDiv.className = '';
                    resultDiv.classList.add(data.sentiment.toLowerCase());
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                loader.style.display = 'none';
                errorDiv.textContent = 'Error connecting to server';
                errorDiv.style.display = 'block';
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>''')
    
    print("Created templates and static directories")

if __name__ == '__main__':
    create_templates()
    print("Starting sentiment analysis web application...")
    print("Open http://127.0.0.1:5000/ in your browser")
    app.run(debug=True) 