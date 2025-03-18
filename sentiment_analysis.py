#!/usr/bin/env python3
# Sentiment Analysis with Logistic Regression

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import parallel_backend
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from scipy.stats import uniform
import time
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Global stemmer instance
porter = PorterStemmer()

# Twitter abbreviations/slang mapping
TWITTER_SLANG = {
    'b4': 'before',
    'bfn': 'bye for now',
    'bgd': 'background',
    'brb': 'be right back',
    'btw': 'by the way',
    'cu': 'see you',
    'cuz': 'because',
    'dm': 'direct message',
    'fb': 'facebook',
    'fml': 'fuck my life',
    'ftw': 'for the win',
    'fw': 'forward',
    'fyi': 'for your information',
    'gtg': 'got to go',
    'idk': 'i do not know',
    'imo': 'in my opinion',
    'imho': 'in my humble opinion',
    'irl': 'in real life',
    'jk': 'just kidding',
    'lmao': 'laughing my ass off',
    'lol': 'laughing out loud',
    'nbd': 'no big deal',
    'nm': 'not much',
    'np': 'no problem',
    'nsfw': 'not safe for work',
    'omg': 'oh my god',
    'omw': 'on my way',
    'ppl': 'people',
    'rofl': 'rolling on the floor laughing',
    'smh': 'shaking my head',
    'tbh': 'to be honest',
    'tbt': 'throwback thursday',
    'tgif': 'thank god it\'s friday',
    'thx': 'thanks',
    'til': 'today i learned',
    'ttyl': 'talk to you later',
    'txt': 'text',
    'u': 'you',
    'ur': 'your',
    'w/e': 'whatever',
    'wtf': 'what the fuck',
    'wtg': 'way to go',
    'wth': 'what the hell',
    'xoxo': 'hugs and kisses',
    'yd': 'yard',
    'yt': 'youtube',
    'y': 'why',
    '2day': 'today',
    '4ward': 'forward',
    'gr8': 'great',
    'b': 'be',
    'c': 'see',
    'd': 'the',
    'e': 'the',
    'f': 'fuck',
    'h8': 'hate',
    'j': 'joke',
    'k': 'okay',
    'l8r': 'later',
    'm': 'am',
    'n': 'and',
    'o': 'oh',
    'r': 'are',
    't': 'the',
    'u': 'you',
    'w': 'with',
    'w/': 'with',
    'w/o': 'without',
    'y': 'why',
    'z': 'the'
}

# Common negation words
NEGATION_WORDS = {
    'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never',
    'hardly', 'scarcely', 'barely', 'doesnt', 'isnt', 'wasnt', 'shouldnt',
    'wouldnt', 'couldnt', 'wont', 'cant', 'dont', 'aint', 'havent', 'hasnt', 'hadnt'
}

# Emoticon mapping to sentiment tokens
EMOTICON_SENTIMENT = {
    ':)': ' positive_emoticon ',
    ':-)': ' positive_emoticon ',
    ';)': ' positive_emoticon ',
    ':D': ' positive_emoticon ',
    '=)': ' positive_emoticon ',
    ':-D': ' positive_emoticon ',
    ':P': ' positive_emoticon ',
    ':-P': ' positive_emoticon ',
    ':(': ' negative_emoticon ',
    ':-(': ' negative_emoticon ',
    ':/': ' negative_emoticon ',
    ':-/': ' negative_emoticon ',
    ':|': ' neutral_emoticon ',
    ':-|': ' neutral_emoticon '
}

# Pre-compiled regex patterns for improved performance
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#(\w+)')
REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{2,}')
HTML_PATTERN = re.compile(r'<[^>]*>')
EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F700-\U0001F77F"  # alchemical symbols
    u"\U0001F780-\U0001F7FF"  # Geometric Shapes
    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U0001FA00-\U0001FA6F"  # Chess Symbols
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    u"\U00002702-\U000027B0"  # Dingbats
    u"\U000024C2-\U0001F251"
    "]+", 
    flags=re.UNICODE
)
EMOTES_PATTERN = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
NUMBERS_PATTERN = re.compile(r'\d+')
PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')

def preprocess_text(text):
    """Preprocess Twitter text for sentiment analysis"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace emoticons with sentiment markers
    for emoticon, replacement in EMOTICON_SENTIMENT.items():
        text = text.replace(emoticon, replacement)
    
    # Normalize URLs, mentions, and hashtags
    text = URL_PATTERN.sub(' url_token ', text)
    text = MENTION_PATTERN.sub(' mention_token ', text)
    hashtags = HASHTAG_PATTERN.findall(text)
    text = HASHTAG_PATTERN.sub(' hashtag_token ', text)
    
    # Remove HTML and normalize repeated characters
    text = HTML_PATTERN.sub('', text)
    text = REPEATED_CHARS_PATTERN.sub(r'\1\1', text)
    
    # Expand Twitter slang and abbreviations
    words = text.split()
    normalized_words = []
    
    for word in words:
        normalized_words.append(TWITTER_SLANG.get(word, word))
    
    text = ' '.join(normalized_words)
    
    # Handle negations by marking subsequent words
    words = text.split()
    result = []
    negation = False
    
    for word in words:
        if word in NEGATION_WORDS or word.endswith("n't"):
            negation = True
            result.append(word)
        elif negation and word.isalnum():
            result.append("NOT_" + word)
        else:
            if not word.isalnum():
                negation = False
            result.append(word)
    
    text = ' '.join(result)
    
    # Handle emojis and numbers
    emojis = EMOJI_PATTERN.findall(text)
    if emojis:
        text = EMOJI_PATTERN.sub(' emoji_token ', text)
    
    text = NUMBERS_PATTERN.sub(' number_token ', text)
    text = PUNCTUATION_PATTERN.sub(' ', text)
    text = WHITESPACE_PATTERN.sub(' ', text)
    
    # Add back hashtag content for sentiment analysis
    if hashtags:
        text += ' ' + ' '.join(hashtags)
    
    return text.strip()

def tokenizer(text):
    """Split text into words"""
    return text.split()

def tokenizer_porter(text):
    """Split text into stemmed words"""
    return [porter.stem(word) for word in text.split()]

def get_class_weights(y):
    """Calculate class weights to handle imbalance"""
    classes = np.unique(y)
    weights = len(y) / (len(classes) * np.bincount(y))
    return {i: weights[i] for i in range(len(weights))}

def load_data(file_path, sample_size=None, balance_classes=True):
    """Load Twitter sentiment data with balanced sampling"""
    print(f"Loading data from {file_path}")
    
    is_large_file = os.path.getsize(file_path) > 1000000
    
    try:
        if is_large_file and sample_size:
            print("Scanning file to find positive and negative examples...")
            
            pos_samples = []
            neg_samples = []
            total_chunks_read = 0
            chunk_size = min(200000, sample_size * 4)
            
            for chunk in pd.read_csv(file_path, 
                                    encoding='latin-1', 
                                    names=['Sentiment', 'ItemID', 'Date', 'Flag', 'User', 'SentimentText'],
                                    chunksize=chunk_size):
                total_chunks_read += 1
                
                chunk_pos = chunk[chunk['Sentiment'] == 4]
                chunk_neg = chunk[chunk['Sentiment'] == 0]
                
                if not chunk_pos.empty and len(pos_samples) < sample_size // 2:
                    pos_samples.append(chunk_pos)
                
                if not chunk_neg.empty and len(neg_samples) < sample_size // 2:
                    neg_samples.append(chunk_neg)
                
                total_pos = sum(len(df) for df in pos_samples)
                total_neg = sum(len(df) for df in neg_samples)
                
                print(f"Chunks read: {total_chunks_read}, " 
                      f"Positive samples: {total_pos}, "
                      f"Negative samples: {total_neg}")
                
                if total_pos >= sample_size // 2 and total_neg >= sample_size // 2:
                    break
                
                if total_chunks_read > 10:
                    break
            
            # Process found samples
            if pos_samples and neg_samples:
                pos_data = pd.concat(pos_samples)
                neg_data = pd.concat(neg_samples)
                
                if balance_classes:
                    target_samples = min(sample_size // 2, len(pos_data), len(neg_data))
                    pos_data = pos_data.sample(target_samples, random_state=42)
                    neg_data = neg_data.sample(target_samples, random_state=42)
                    data = pd.concat([pos_data, neg_data])
                else:
                    data = pd.concat([pos_data, neg_data])
                    data = data.sample(min(sample_size, len(data)), random_state=42)
            else:
                raise ValueError("Could not find examples of both sentiment classes in the data")
        else:
            # For smaller files
            data = pd.read_csv(file_path, 
                            encoding='latin-1', 
                            names=['Sentiment', 'ItemID', 'Date', 'Flag', 'User', 'SentimentText'])
            
            if 0 not in data['Sentiment'].values or 4 not in data['Sentiment'].values:
                raise ValueError(f"Data does not contain both sentiment classes (0 and 4)")
                
            if sample_size and balance_classes:
                pos_samples = data[data['Sentiment'] == 4].sample(
                    min(sample_size // 2, len(data[data['Sentiment'] == 4])), 
                    random_state=42
                )
                neg_samples = data[data['Sentiment'] == 0].sample(
                    min(sample_size // 2, len(data[data['Sentiment'] == 0])), 
                    random_state=42
                )
                data = pd.concat([pos_samples, neg_samples])
            elif sample_size:
                data = data.sample(min(sample_size, len(data)), random_state=42)
        
        # Track and convert class distribution
        print("Class distribution before binary conversion:")
        print(data['Sentiment'].value_counts())
        
        data['Sentiment'] = data['Sentiment'].apply(lambda x: 1 if x == 4 else 0)
        
        print(f"Loaded {len(data)} records with sentiment distribution:")
        print(data['Sentiment'].value_counts())
        
        if 0 not in data['Sentiment'].values or 1 not in data['Sentiment'].values:
            raise ValueError(f"Final data does not contain both binary sentiment classes (0 and 1)")
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def train_sentiment_model(train_path, test_path=None, 
                          sample_size=None,
                          max_features=30000, 
                          n_iter=10,
                          output_model_path='sentiment_model.pkl',
                          calibrate=True):
    """Train a logistic regression model for Twitter sentiment analysis"""
    start_time = time.time()
    
    # Load and prepare data
    train_data = load_data(train_path, sample_size, balance_classes=True)
    
    X = train_data['SentimentText']
    y = train_data['Sentiment']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data size: {X_train.shape[0]}, Validation data size: {X_val.shape[0]}")
    
    # Configure TF-IDF and model
    stop = stopwords.words('english')
    
    tfidf = TfidfVectorizer(
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        max_features=max_features,
        min_df=5,  # Minimum document frequency
        max_df=0.8  # Maximum document frequency
    )
    
    class_weights = get_class_weights(y_train)
    print("Using class weights:", class_weights)
    
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='saga',
        tol=1e-4,
        n_jobs=-1,
        class_weight=class_weights
    )
    
    pipeline = Pipeline([
        ('vect', tfidf),
        ('clf', lr)
    ])
    
    # Define parameter search spaces
    l1_l2_params = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'vect__preprocessor': [None, preprocess_text],
        'vect__norm': ['l1', 'l2'],
        'clf__C': uniform(0.1, 10.0),
        'clf__penalty': ['l1', 'l2']
    }
    
    elasticnet_params = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'vect__preprocessor': [None, preprocess_text],
        'vect__norm': ['l1', 'l2'],
        'clf__C': uniform(0.1, 10.0),
        'clf__penalty': ['elasticnet'],
        'clf__l1_ratio': uniform(0, 1)
    }
    
    # Allocate iterations between parameter sets
    total_iter = n_iter
    l1_l2_iter = int(total_iter * 0.7)
    elasticnet_iter = total_iter - l1_l2_iter
    
    print(f"Starting randomized search with {n_iter} total iterations...")
    
    # Search with L1/L2 penalties
    print(f"Searching with L1/L2 penalties ({l1_l2_iter} iterations)...")
    rs_l1_l2 = RandomizedSearchCV(
        pipeline,
        param_distributions=l1_l2_params,
        n_iter=l1_l2_iter,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    with parallel_backend('threading', n_jobs=-1):
        rs_l1_l2.fit(X_train, y_train)
    
    # Search with Elasticnet if iterations allocated
    if elasticnet_iter > 0:
        print(f"Searching with Elasticnet penalties ({elasticnet_iter} iterations)...")
        rs_elasticnet = RandomizedSearchCV(
            pipeline,
            param_distributions=elasticnet_params,
            n_iter=elasticnet_iter,
            scoring='accuracy',
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=43
        )
        
        with parallel_backend('threading', n_jobs=-1):
            rs_elasticnet.fit(X_train, y_train)
        
        # Select the better model
        if rs_elasticnet.best_score_ > rs_l1_l2.best_score_:
            random_search = rs_elasticnet
        else:
            random_search = rs_l1_l2
    else:
        random_search = rs_l1_l2
    
    # Get best classifier
    clf = random_search.best_estimator_
    
    # Calibrate probabilities if requested
    if calibrate:
        print("Calibrating probability estimates...")
        clf_calibrated = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
        clf_calibrated.fit(X_val, y_val)
        clf = clf_calibrated
    
    # Evaluation
    print('Best parameter set:', random_search.best_params_)
    print(f'Best cross-validation accuracy: {random_search.best_score_:.3f}')
    
    val_preds = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f'Validation accuracy: {val_accuracy:.3f}')
    print('\nValidation Classification Report:')
    print(classification_report(y_val, val_preds))
    
    # Test set evaluation if provided
    if test_path:
        test_data = load_data(test_path)
        X_test = test_data['SentimentText']
        y_test = test_data['Sentiment']
        
        test_preds = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)
        print(f'Test accuracy: {test_accuracy:.3f}')
        print('\nTest Classification Report:')
        print(classification_report(y_test, test_preds))
    
    # Save model
    print(f"Saving model to {output_model_path}")
    pickle.dump(clf, open(output_model_path, 'wb'), protocol=4)
    
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    return clf

def predict_sentiment(text, model_path='sentiment_model.pkl', return_proba=False):
    """Predict sentiment for text with optional confidence scores"""
    clf = pickle.load(open(model_path, 'rb'))
    
    if isinstance(text, str):
        text = [text]
    
    # Extract preprocessor from model pipeline
    if hasattr(clf, 'named_steps'):
        preprocess_fn = clf.named_steps['vect'].preprocessor
    elif hasattr(clf, 'estimator') and hasattr(clf.estimator, 'named_steps'):
        preprocess_fn = clf.estimator.named_steps['vect'].preprocessor
    else:
        preprocess_fn = None
    
    # Apply preprocessing if model was trained with it
    if preprocess_fn is not None:
        text = [preprocess_fn(t) for t in text]
    
    if return_proba:
        return clf.predict_proba(text)[:, 1]
    else:
        return clf.predict(text)

if __name__ == "__main__":
    # Data file paths
    train_path = os.path.join('sentiment_data', 'training.1600000.processed.noemoticon.csv')
    test_path = os.path.join('sentiment_data', 'testdata.manual.2009.06.14.csv')
    
    # Training configuration
    sample_size = 20000  # Use at least 50,000 for production
    
    # Train model
    model = train_sentiment_model(
        train_path, 
        test_path,
        sample_size=sample_size,
        max_features=20000,
        n_iter=8,
        calibrate=True
    )
    
    # Example predictions
    example_texts = [
        "This is really bad, I don't like it at all",
        "I love this!",
        ":)",
        "I'm sad... :(",
        "The product works exactly as described. Very happy with my purchase!",
        "Customer service was terrible and the product broke after two days",
        "Just tried the new iPhone and it's amazing! #technology #apple",
        "@friend Check out this cool new app I found! https://example.com"
    ]
    
    # Get predictions
    predictions = predict_sentiment(example_texts, 'sentiment_model.pkl', return_proba=False)
    proba = predict_sentiment(example_texts, 'sentiment_model.pkl', return_proba=True)
    
    print("\nSample predictions:")
    for text, pred, prob in zip(example_texts, predictions, proba):
        sentiment = "positive" if pred == 1 else "negative"
        prob_display = prob if pred == 1 else 1 - prob  # Show confidence in prediction
        print(f"{text} --> {sentiment} (confidence: {prob_display:.2f})") 