# Trust by Design
### Cross-Platform Content Intelligence Tool
*Human-Centered Data Science — Final Project*
*Akanksha Bharambe & Rohit Yadav — UIUC*

---

## Overview

Trust by Design is an AI-powered tool that predicts how a news headline
will perform across Facebook, LinkedIn, and Google+ — and evaluates its
credibility score.

Built on 93,000+ real news articles from the UCI News Popularity Dataset,
the system uses platform-specific machine learning models to show that
content success is not universal — it depends on platform culture,
audience expectations, and content framing.

---

## Live Demo

 [Launch the app](your-huggingface-link-here)

---

## Key Features

- **Platform Virality Prediction** — separate ML models for Facebook, LinkedIn, and Google+
- **Credibility Score** — engineered trust metric based on linguistic signals
- **Explainable Recommendations** — platform-specific reasoning for every prediction
- **Headline Optimization Tips** — actionable, dynamic suggestions per platform
- **Sentiment Analysis** — automatic tone detection using TextBlob

---

## Dataset

**UCI News Popularity in Multiple Social Media Platforms**
- 93,239 news articles
- Platforms: Facebook, LinkedIn, Google+
- Features: title, headline, topic, sentiment scores, share counts
- Link: https://archive.ics.uci.edu/dataset/432/news+popularity+in+multiple+social-media+platforms

---

## ML Models

Three separate Gradient Boosting Classifiers trained per platform:

| Platform | AUC-ROC | Accuracy |
|----------|---------|----------|
| Facebook | 0.7889 | 80% |
| LinkedIn | 0.7400 | 75% |
| Google+  | 0.7388 | 72% |

**Features used:**
- TF-IDF on title + headline (3,000 features, bigrams)
- Structural features: title length, word count, punctuation, caps ratio
- Sentiment: polarity scores for title and headline
- Topic category encoding

**Label definition:** Top 25% share count per platform = viral (1)

---

## Project Structure

```text
trust-by-design/
├── app.py                  — Streamlit UI
├── inference.py            — ML prediction logic
├── trust_score.py          — Credibility scoring engine
├── requirements.txt
├── README.md
│
├── models/
│   ├── facebook_model.pkl
│   ├── linkedin_model.pkl
│   ├── googleplus_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_meta.pkl
│
├── notebooks/
│   └── EDA_and_Training.ipynb
│
├── data/
│   └── .gitkeep
│
├── documentation/
│   ├── EDA_and_Training.ipynb.pdf
│   ├── Project Abstract.pdf
│   └── Project Proposal.pdf
│
└── presentation/
    ├── Trust_by_Design_Group_2.pptx
    └── Trust by Design Midterm Project Report.pdf
```  

---

## Run Locally

```bash
git clone https://huggingface.co/spaces/your-username/trust-by-design
cd trust-by-design
pip install -r requirements.txt
streamlit run app.py
```

---

## Methodology

1. **Framing** — We treat social sharing as an engagement signal, not a
   direct proxy for trust. Platform culture shapes what content gets
   amplified, which reflects perceived credibility and relevance.

2. **Distribution shift** — Models trained on one platform do not
   generalize to another, confirming that virality is platform-specific.

3. **Trust scoring** — Engineered from linguistic features: capitalization,
   punctuation, professional vocabulary, clickbait patterns, and
   headline structure.

4. **Human-centered design** — Every prediction includes an explanation
   and actionable recommendations, making the system useful for real
   content creators.

---

## Future Work

- Multimodal expansion: image thumbnails, visual credibility signals
- Real-time news API integration
- User feedback loop to improve trust scoring
- Cross-lingual support

---

## Citation

Moniz, N. & Torgo, L. (2018). Multi-Source Social Feedback of Online
News Feeds. UCI Machine Learning Repository.
https://doi.org/10.24432/C5PG8S

