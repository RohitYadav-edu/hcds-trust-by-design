# inference.py
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Load models and artifacts ─────────────────────────────────────────────────
def load_artifacts(model_dir='models'):
    artifacts = {}
    files = {
        'facebook':   f'{model_dir}/facebook_model.pkl',
        'linkedin':   f'{model_dir}/linkedin_model.pkl',
        'googleplus': f'{model_dir}/googleplus_model.pkl',
        'tfidf':      f'{model_dir}/tfidf_vectorizer.pkl',
        'meta':       f'{model_dir}/model_meta.pkl',
    }
    for key, path in files.items():
        with open(path, 'rb') as f:
            artifacts[key] = pickle.load(f)
    return artifacts

# ── Build structural features for a single input ──────────────────────────────
def build_struct_features(title, headline, topic, sentiment_title, 
                           sentiment_headline, meta):
    feats = {}
    text = title + ' ' + headline

    feats['title_len']          = len(title)
    feats['title_words']        = len(title.split())
    feats['headline_words']     = len(headline.split())
    feats['exclamation']        = title.count('!')
    feats['question']           = title.count('?')
    feats['caps_ratio']         = sum(1 for c in title if c.isupper()) / max(len(title), 1)
    feats['num_count']          = sum(1 for c in title if c.isdigit())
    feats['colon_present']      = int(':' in title)
    feats['quote_present']      = int('"' in title)
    feats['sentiment_title']    = sentiment_title
    feats['sentiment_headline'] = sentiment_headline
    feats['sentiment_diff']     = abs(sentiment_title - sentiment_headline)

    # Topic dummies — must match training columns exactly
    for col in meta['topic_columns']:
        topic_val = col.replace('topic_', '')
        feats[col] = 1 if topic == topic_val else 0

    # Build in correct column order
    row = [feats.get(col, 0) for col in meta['struct_columns']]
    return np.array(row, dtype=np.float32).reshape(1, -1)

# ── Main prediction function ──────────────────────────────────────────────────
def predict(title, headline, topic, sentiment_title, sentiment_headline, artifacts):
    meta   = artifacts['meta']
    tfidf  = artifacts['tfidf']

    # TF-IDF on combined text
    text        = title + ' ' + headline
    tfidf_vec   = tfidf.transform([text])

    # Structural features
    struct_vec  = build_struct_features(
                    title, headline, topic,
                    sentiment_title, sentiment_headline, meta)

    # Combined feature vector
    X = hstack([tfidf_vec, csr_matrix(struct_vec)])

    # Predict for each platform
    results = {}
    for platform in ['facebook', 'linkedin', 'googleplus']:
        model = artifacts[platform]
        prob  = model.predict_proba(X)[0][1]  # probability of viral
        results[platform] = round(float(prob) * 100, 1)

    return results