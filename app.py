import streamlit as st
from textblob import TextBlob
from inference import load_artifacts, predict
from trust_score import compute_trust_score

st.set_page_config(
    page_title="Trust by Design",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1b2a; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3, .stMarkdown p { color: #f5f0e8; }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: #1b2d3e !important;
        color: #f5f0e8 !important;
        border: 1px solid #c9a84c !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #c9a84c, #e8c96d);
        color: #0d1b2a;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
    [data-testid="stMetric"] {
        background-color: #1b2d3e;
        border: 1px solid #c9a84c;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: #a89070 !important; }
    [data-testid="stMetricValue"] { color: #f5f0e8 !important; }
    .stProgress > div > div > div { background-color: #c9a84c !important; }
    .streamlit-expanderHeader {
        background-color: #1b2d3e !important;
        border: 1px solid #c9a84c !important;
        border-radius: 8px !important;
        color: #f5f0e8 !important;
    }
    .streamlit-expanderContent {
        background-color: #1b2d3e !important;
        border: 1px solid #c9a84c !important;
        border-top: none !important;
        color: #a89070 !important;
    }
    hr { border-color: #c9a84c !important; opacity: 0.3; }
    .stSuccess { background-color: #1b2d1b !important; color: #a8d5a2 !important; border: 1px solid #5a9e5a !important; }
    .stInfo    { background-color: #1b2d3e !important; color: #a89070 !important; border: 1px solid #c9a84c !important; }
    .stError   { background-color: #2e1a1a !important; color: #e07070 !important; border: 1px solid #c0504d !important; }
    .stCaption { color: #c9a84c !important; }
    label { color: #a89070 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_artifacts():
    return load_artifacts('models')

artifacts = get_artifacts()
meta      = artifacts['meta']

# ── Topic labels ──────────────────────────────────────────────────────────────
TOPIC_LABELS = {
    'obama':     'Politics',
    'economy':   'Economy',
    'microsoft': 'Technology',
    'palestine': 'Conflict'
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 1rem 0 0.5rem;'>
    <span style='background:#c9a84c; color:#0d1b2a; font-size:12px;
                 font-weight:600; padding:4px 14px; border-radius:20px;
                 letter-spacing:0.08em;'>
        HUMAN-CENTERED DATA SCIENCE
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; font-size:2.4rem; font-weight:700;
           color:#f5f0e8; margin-top:0.5rem;'>
    🔍 Trust by Design
</h1>
<p style='text-align:center; color:#a89070; font-size:1rem; margin-top:-0.5rem;'>
    Cross-Platform Content Intelligence Tool
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown("#### Enter Your Content")

title = st.text_input(
    "Headline (required)",
    placeholder="e.g. New Report: AI Investment in Enterprise Sector Grows 40%"
)

headline = st.text_area(
    "Article Summary (optional — improves prediction accuracy)",
    placeholder="Paste a short excerpt or summary of the article here...",
    height=120
)

selected_label = st.selectbox(
    "Topic Category",
    options=list(TOPIC_LABELS.values())
)
topic = [k for k, v in TOPIC_LABELS.items() if v == selected_label][0]

st.divider()

# ── Platform explanation logic ────────────────────────────────────────────────
def explain_platform(platform, title, topic, sentiment_title, scores):
    reasons = []

    if platform == 'facebook':
        if topic == 'obama':
            reasons.append("Political content consistently drives higher engagement on Facebook")
        if topic == 'palestine':
            reasons.append("Conflict and social justice topics generate strong reactions on Facebook")
        if sentiment_title < -0.05:
            reasons.append("Negative sentiment tends to increase sharing behavior on Facebook")
        if sentiment_title > 0.05:
            reasons.append("Positive emotional tone resonates with Facebook's broad audience")
        if title.count('!') > 0:
            reasons.append("Exclamatory language drives click-through on Facebook")
        if len(title.split()) <= 10:
            reasons.append("Short punchy headlines perform well in Facebook feeds")
        if not reasons:
            reasons.append("Broad general appeal and accessible language suit Facebook's audience")

    elif platform == 'linkedin':
        if topic == 'microsoft':
            reasons.append("Technology and industry news is highly valued on LinkedIn")
        if topic == 'economy':
            reasons.append("Economic insight and business analysis drives LinkedIn engagement")
        if -0.05 <= sentiment_title <= 0.05:
            reasons.append("Neutral, factual tone aligns with LinkedIn's professional culture")
        if ':' in title:
            reasons.append("Structured headline format signals professionalism on LinkedIn")
        if any(c.isdigit() for c in title):
            reasons.append("Data points and statistics increase credibility on LinkedIn")
        if not reasons:
            reasons.append("Professional framing and informational tone suit LinkedIn audiences")

    elif platform == 'googleplus':
        if topic == 'microsoft':
            reasons.append("Tech-focused content performed strongly in Google+ communities")
        if topic == 'economy':
            reasons.append("Analytical economic content suits Google+'s informed user base")
        if -0.05 <= sentiment_title <= 0.05:
            reasons.append("Neutral informational tone aligns with Google+'s niche communities")
        if not reasons:
            reasons.append("Informational tone and niche appeal suit Google+'s audience")

    return reasons

# ── Optimization tips ─────────────────────────────────────────────────────────
def get_optimization_tips(platform, title, topic, sentiment_title, trust):
    tips = []

    if platform == 'facebook':
        if -0.05 <= sentiment_title <= 0.05:
            tips.append("Add emotional language — neutral headlines tend to underperform on Facebook")
        if topic in ['economy', 'microsoft']:
            tips.append("Add a human angle — 'How this affects you' style works better on Facebook")
        if ':' in title:
            tips.append("Consider removing the colon — conversational headlines do better here")
        if len(title.split()) > 12:
            tips.append("Shorten your headline — Facebook feeds favor punchy, concise titles")
        if not tips:
            tips.append("Try starting with 'Why...' or 'How...' to boost Facebook shareability")
        tips.append("Use strong action verbs and storytelling hooks")

    elif platform == 'linkedin':
        if sentiment_title < -0.05:
            tips.append("Reduce negative sentiment — LinkedIn audiences prefer constructive framing")
        if topic in ['obama', 'palestine']:
            tips.append("Add a professional or policy angle to make it relevant to LinkedIn readers")
        if not any(c.isdigit() for c in title):
            tips.append("Add a statistic or data point — numbers boost LinkedIn credibility")
        if ':' not in title:
            tips.append("Use colon structure e.g. 'Topic: Key Finding' for professional framing")
        if not tips:
            tips.append("Lead with data, insight, or a bold industry claim")
        tips.append("Keep tone analytical — avoid emotional or sensational language")

    elif platform == 'googleplus':
        if sentiment_title < -0.05:
            tips.append("Soften negative tone — Google+ communities preferred informational content")
        if topic in ['obama', 'palestine']:
            tips.append("Frame around policy or analysis rather than emotion for Google+ audiences")
        if not tips:
            tips.append("Use clear, informational language suited to niche topic communities")
        tips.append("Concise and factual headlines performed best on Google+")

    return tips

# ── Analyze button ────────────────────────────────────────────────────────────
if st.button("🚀 Analyze Content"):

    if not title.strip():
        st.error("Please enter a headline to analyze.")
    else:
        headline_text = headline.strip() if headline.strip() else title

        blob_title         = TextBlob(title)
        blob_headline      = TextBlob(headline_text)
        sentiment_title    = blob_title.sentiment.polarity
        sentiment_headline = blob_headline.sentiment.polarity

        with st.spinner("Analyzing content across platforms..."):
            scores = predict(
                title, headline_text, topic,
                sentiment_title, sentiment_headline,
                artifacts
            )
            trust = compute_trust_score(title, headline_text)

        st.success("✅ Analysis complete!")
        st.divider()

        # ── Virality Scores ───────────────────────────────────────────────────
        st.markdown("### 📊 Platform Virality Probability")
        st.caption("Likelihood this content reaches top 25% engagement on each platform")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ⓕ Facebook", f"{scores['facebook']}%")
            st.progress(scores['facebook'] / 100)
        with col2:
            st.metric("[in] LinkedIn", f"{scores['linkedin']}%")
            st.progress(scores['linkedin'] / 100)
        with col3:
            st.metric("𝐆 Google+", f"{scores['googleplus']}%")
            st.progress(scores['googleplus'] / 100)

        st.divider()

        # ── Best Platform + Explanation ───────────────────────────────────────
        best = max(scores, key=scores.get)
        platform_labels = {
            'facebook':   'ⓕ Facebook',
            'linkedin':   '[in] LinkedIn',
            'googleplus': '𝐆 Google+'
        }

        st.markdown("### 🏆 Best Platform Recommendation")
        st.success(
            f"**{platform_labels[best]}** — highest predicted engagement at **{scores[best]}%**"
        )

        st.markdown("**Why this platform?**")
        reasons = explain_platform(best, title, topic, sentiment_title, scores)
        for r in reasons:
            st.markdown(f"- {r}")

        st.divider()

        # ── Trust Score ───────────────────────────────────────────────────────
        st.markdown("### 🛡️ Credibility Score")

        trust_col1, trust_col2 = st.columns([1, 2])
        with trust_col1:
            color_map = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}
            st.metric("Score", f"{trust['score']} / 100")
            st.markdown(f"**{color_map[trust['label']]} {trust['label']} Credibility**")
            st.progress(trust['score'] / 100)

        with trust_col2:
            if trust['boosts']:
                st.markdown("**Positive signals:**")
                for b in trust['boosts']:
                    st.markdown(
                        f"<span style='color:#a8d5a2'>{b}</span>",
                        unsafe_allow_html=True)
            if trust['flags']:
                st.markdown("**Areas to improve:**")
                for f in trust['flags']:
                    st.markdown(
                        f"<span style='color:#e8c96d'>{f}</span>",
                        unsafe_allow_html=True)

        st.divider()

        # ── Sentiment ─────────────────────────────────────────────────────────
        st.markdown("### 🎭 Detected Sentiment")

        def sentiment_label(score):
            if score > 0.05:  return f"Positive ({score:.3f})"
            if score < -0.05: return f"Negative ({score:.3f})"
            return f"Neutral ({score:.3f})"

        s1, s2 = st.columns(2)
        with s1:
            st.metric("Headline Sentiment", sentiment_label(sentiment_title))
        with s2:
            st.metric("Summary Sentiment", sentiment_label(sentiment_headline))

        st.divider()

        # ── Optimization Tips ─────────────────────────────────────────────────
        st.markdown("### ✏️ Headline Optimization Tips")

        with st.expander("📘 Optimize for Facebook"):
            tips = get_optimization_tips('facebook', title, topic, sentiment_title, trust)
            for t in tips:
                st.markdown(f"- {t}")

        with st.expander("💼 Optimize for LinkedIn"):
            tips = get_optimization_tips('linkedin', title, topic, sentiment_title, trust)
            for t in tips:
                st.markdown(f"- {t}")

        with st.expander("🛡️ Optimize for Credibility"):
            st.markdown(
                "- Avoid ALL CAPS, excessive punctuation, and clickbait phrases\n"
                "- Use colon structure — e.g. 'Topic: Key Finding'\n"
                "- Keep headline between 6 and 14 words\n"
                "- Use neutral, factual language"
            )
            st.markdown(f"**Trust score:** {trust['score']}/100 ({trust['label']})")

        st.divider()
        st.caption(
            "Trust by Design · Human-Centered Data Science · "
            "Models trained on UCI News Popularity Dataset (93k+ articles)"
        )