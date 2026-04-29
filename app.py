import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from inference import load_artifacts, predict
from trust_score import compute_trust_score

st.set_page_config(
    page_title="Trust by Design",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1b2a; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
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
    .stWarning { background-color: #2e2a1a !important; color: #e8c96d !important; border: 1px solid #c9a84c !important; }
    .stCaption { color: #c9a84c !important; }
    label { color: #a89070 !important; }

    /* Centered and styled tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1b2d3e;
        border-radius: 12px;
        padding: 4px;
        display: flex;
        justify-content: center;
        gap: 4px;
        width: fit-content;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a89070;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-size: 15px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #c9a84c !important;
        color: #0d1b2a !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_artifacts():
    return load_artifacts('models')

@st.cache_data
def load_dataset():
    df = pd.read_csv('data/News_Final.csv')
    df['Title']    = df['Title'].fillna('')
    df['Headline'] = df['Headline'].fillna('')
    df['Topic']    = df['Topic'].fillna('unknown')
    df['text']     = df['Title'] + ' ' + df['Headline']
    df = df[~((df['Facebook'] == -1) & (df['GooglePlus'] == -1) & (df['LinkedIn'] == -1))]

    def quick_trust(row):
        title = str(row['Title'])
        caps_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
        score = 60
        if caps_ratio > 0.4:          score -= 15
        if title.count('!') >= 2:     score -= 10
        if title.count('!') == 1:     score -= 5
        if title.endswith('?'):       score -= 8
        if len(title.split()) < 4:    score -= 8
        if len(title.split()) > 18:   score -= 5
        prof_words = ["analysis","report","study","research","data","strategy",
                      "policy","growth","investment","economy","market","technology"]
        hits = sum(1 for w in prof_words if w in title.lower())
        if hits >= 2:                 score += 15
        elif hits == 1:               score += 7
        if ':' in title:              score += 8
        if 6 <= len(title.split()) <= 14: score += 10
        if caps_ratio < 0.15:         score += 7
        return max(0, min(100, score))

    df['trust_score'] = df.apply(quick_trust, axis=1)

    for col in ['Facebook', 'LinkedIn', 'GooglePlus']:
        valid = df[df[col] >= 0][col]
        thresh = np.percentile(valid, 75)
        df[f'{col}_viral'] = ((df[col] >= thresh) & (df[col] >= 0)).astype(int)
        df[f'{col}_norm']  = df[col].clip(lower=0)
        max_val = df[f'{col}_norm'].max()
        df[f'{col}_score'] = (df[f'{col}_norm'] / max_val * 100).fillna(0)

    return df

artifacts = get_artifacts()
meta      = artifacts['meta']

TOPIC_LABELS = {
    'obama':     'Politics',
    'economy':   'Economy',
    'microsoft': 'Technology',
    'palestine': 'Conflict'
}

PLOT_THEME = {
    'paper_bgcolor': '#0d1b2a',
    'plot_bgcolor':  '#1b2d3e',
    'font_color':    '#f5f0e8',
    'gridcolor':     '#2a3f54',
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
<h1 style='text-align:center; font-size:2.4rem; font-weight:700;
           color:#f5f0e8; margin-top:0.5rem;'>
    🔍 Trust by Design
</h1>
<p style='text-align:center; color:#a89070; font-size:1rem; margin-top:-0.5rem;'>
    Cross-Platform Content Intelligence Tool
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Helper functions ──────────────────────────────────────────────────────────
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

def sentiment_label(score):
    if score > 0.05:  return f"Positive ({score:.3f})"
    if score < -0.05: return f"Negative ({score:.3f})"
    return f"Neutral ({score:.3f})"

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Predict & Analyze",
    "Trust vs Virality",
    "Platform Bias Explorer",
    "Credibility Audit"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict & Analyze
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
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

    if st.button("🚀 Analyze Content", key="analyze"):
        if not title.strip():
            st.error("Please enter a headline to analyze.")
        else:
            headline_text      = headline.strip() if headline.strip() else title
            sentiment_title    = TextBlob(title).sentiment.polarity
            sentiment_headline = TextBlob(headline_text).sentiment.polarity

            with st.spinner("Analyzing content across platforms..."):
                scores = predict(title, headline_text, topic,
                                 sentiment_title, sentiment_headline, artifacts)
                trust  = compute_trust_score(title, headline_text)

            st.success("✅ Analysis complete!")
            st.divider()

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

            best = max(scores, key=scores.get)
            platform_labels = {
                'facebook':   'ⓕ Facebook',
                'linkedin':   '[in] LinkedIn',
                'googleplus': '𝐆 Google+'
            }

            st.markdown("### 🏆 Best Platform Recommendation")
            st.success(f"**{platform_labels[best]}** — highest predicted engagement at **{scores[best]}%**")
            st.markdown("**Why this platform?**")
            for r in explain_platform(best, title, topic, sentiment_title, scores):
                st.markdown(f"- {r}")

            st.divider()

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
                        st.markdown(f"<span style='color:#a8d5a2'>{b}</span>", unsafe_allow_html=True)
                if trust['flags']:
                    st.markdown("**Areas to improve:**")
                    for f in trust['flags']:
                        st.markdown(f"<span style='color:#e8c96d'>{f}</span>", unsafe_allow_html=True)

            st.divider()

            st.markdown("### 🎭 Detected Sentiment")
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Headline Sentiment", sentiment_label(sentiment_title))
            with s2:
                st.metric("Summary Sentiment", sentiment_label(sentiment_headline))

            st.divider()

            st.markdown("### ✏️ Headline Optimization Tips")
            with st.expander("ⓕ Optimize for Facebook"):
                for t in get_optimization_tips('facebook', title, topic, sentiment_title, trust):
                    st.markdown(f"- {t}")
            with st.expander("[in] Optimize for LinkedIn"):
                for t in get_optimization_tips('linkedin', title, topic, sentiment_title, trust):
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
            st.caption("Trust by Design · Human-Centered Data Science · Models trained on UCI News Popularity Dataset (93k+ articles)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trust vs Virality Gap
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Trust vs Virality Gap")
    st.markdown("High engagement does not always mean high credibility. This tab explores the gap between what gets shared and what deserves to be trusted.")
    st.divider()

    with st.spinner("Loading dataset..."):
        df = load_dataset()

    platform_choice = st.selectbox(
        "Select platform to analyze",
        options=["Facebook", "LinkedIn", "Google+"],
        key="tvp_platform"
    )
    topic_filter = st.selectbox(
        "Filter by topic",
        options=["All topics"] + list(TOPIC_LABELS.values()),
        key="tvp_topic"
    )

    col_map = {"Facebook": "Facebook", "LinkedIn": "LinkedIn", "Google+": "GooglePlus"}
    col     = col_map[platform_choice]

    plot_df = df[df[col] >= 0].copy()
    if topic_filter != "All topics":
        topic_key = [k for k, v in TOPIC_LABELS.items() if v == topic_filter][0]
        plot_df   = plot_df[plot_df['Topic'] == topic_key]

    if len(plot_df) > 3000:
        plot_df = plot_df.sample(3000, random_state=42)

    median_trust    = plot_df['trust_score'].median()
    median_virality = plot_df[f'{col}_score'].median()

    def quadrant_label(row):
        high_trust    = row['trust_score']   >= median_trust
        high_virality = row[f'{col}_score'] >= median_virality
        if high_trust and high_virality:     return "Ideal — Credible & Viral"
        if not high_trust and high_virality: return "Viral but Low Credibility"
        if high_trust and not high_virality: return "Credible but Low Reach"
        return "Low Trust & Low Reach"

    plot_df['quadrant'] = plot_df.apply(quadrant_label, axis=1)

    color_map_q = {
        "Ideal — Credible & Viral":      "#a8d5a2",
        "Viral but Low Credibility":     "#e07070",
        "Credible but Low Reach":        "#e8c96d",
        "Low Trust & Low Reach":         "#a89070"
    }

    fig = px.scatter(
        plot_df,
        x='trust_score',
        y=f'{col}_score',
        color='quadrant',
        color_discrete_map=color_map_q,
        hover_data=['Title', 'Topic'],
        labels={
            'trust_score':   'Credibility Score',
            f'{col}_score':  f'{platform_choice} Virality Score',
            'quadrant':      'Content Quadrant'
        },
        title=f'Trust vs Virality — {platform_choice}'
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.add_hline(y=median_virality, line_dash="dash", line_color="#c9a84c", opacity=0.5)
    fig.add_vline(x=median_trust,    line_dash="dash", line_color="#c9a84c", opacity=0.5)
    fig.update_layout(
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
        font=dict(color=PLOT_THEME['font_color']),
        xaxis=dict(gridcolor=PLOT_THEME['gridcolor']),
        yaxis=dict(gridcolor=PLOT_THEME['gridcolor']),
        legend=dict(bgcolor='#1b2d3e', bordercolor='#c9a84c'),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Quadrant Breakdown")
    q_counts = plot_df['quadrant'].value_counts()
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    with q_col1:
        st.metric("Ideal", q_counts.get("Ideal — Credible & Viral", 0))
        st.caption("Credible & Viral")
    with q_col2:
        st.metric("Risky", q_counts.get("Viral but Low Credibility", 0))
        st.caption("Viral but Low Credibility")
    with q_col3:
        st.metric("Underrated", q_counts.get("Credible but Low Reach", 0))
        st.caption("Credible but Low Reach")
    with q_col4:
        st.metric("Poor", q_counts.get("Low Trust & Low Reach", 0))
        st.caption("Low Trust & Low Reach")

    st.divider()
    st.info(
        "The gap between the 'Viral but Low Credibility' and 'Credible but Low Reach' quadrants "
        "reveals how platform culture rewards engagement over accuracy. "
        "Content optimized for virality often sacrifices credibility signals."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Platform Bias Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Platform Bias Explorer")
    st.markdown("The same content is judged differently by each platform. This reveals how platform culture shapes what gets amplified — and what gets ignored.")
    st.divider()

    with st.spinner("Loading dataset..."):
        df = load_dataset()

    bias_topic = st.selectbox(
        "Select topic to explore",
        options=list(TOPIC_LABELS.values()),
        key="bias_topic"
    )
    topic_key = [k for k, v in TOPIC_LABELS.items() if v == bias_topic][0]
    topic_df  = df[df['Topic'] == topic_key].copy()

    fb_avg = topic_df[topic_df['Facebook'] >= 0]['Facebook_score'].mean()
    li_avg = topic_df[topic_df['LinkedIn'] >= 0]['LinkedIn_score'].mean()
    gp_avg = topic_df[topic_df['GooglePlus'] >= 0]['GooglePlus_score'].mean()

    st.markdown(f"#### How {bias_topic} content performs across platforms")

    bar_fig = go.Figure(data=[
        go.Bar(
            x=["Facebook", "LinkedIn", "Google+"],
            y=[fb_avg, li_avg, gp_avg],
            marker_color=["#3b5998", "#0077b5", "#dd4b39"],
            text=[f"{fb_avg:.1f}", f"{li_avg:.1f}", f"{gp_avg:.1f}"],
            textposition='outside',
            textfont=dict(color='#f5f0e8')
        )
    ])
    bar_fig.update_layout(
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
        font=dict(color=PLOT_THEME['font_color']),
        yaxis=dict(gridcolor=PLOT_THEME['gridcolor'], title="Average Virality Score"),
        xaxis=dict(title="Platform"),
        height=400,
        showlegend=False
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("#### Sentiment Distribution by Platform Performance")
    st.caption("Do high-performing articles on each platform share a common sentiment pattern?")

    sent_col1, sent_col2, sent_col3 = st.columns(3)

    for col_name, label, container in [
        ("Facebook",   "Facebook",  sent_col1),
        ("LinkedIn",   "LinkedIn",  sent_col2),
        ("GooglePlus", "Google+",   sent_col3)
    ]:
        with container:
            plat_df  = topic_df[topic_df[col_name] >= 0].copy()
            thresh   = np.percentile(plat_df[col_name], 75)
            viral    = plat_df[plat_df[col_name] >= thresh]['SentimentTitle']
            nonviral = plat_df[plat_df[col_name] <  thresh]['SentimentTitle']

            fig_s = go.Figure()
            fig_s.add_trace(go.Histogram(
                x=viral, name="High Engagement",
                marker_color="#a8d5a2", opacity=0.75, nbinsx=20))
            fig_s.add_trace(go.Histogram(
                x=nonviral, name="Low Engagement",
                marker_color="#e07070", opacity=0.75, nbinsx=20))
            fig_s.update_layout(
                title=label,
                barmode='overlay',
                paper_bgcolor=PLOT_THEME['paper_bgcolor'],
                plot_bgcolor=PLOT_THEME['plot_bgcolor'],
                font=dict(color=PLOT_THEME['font_color'], size=11),
                xaxis=dict(gridcolor=PLOT_THEME['gridcolor'], title="Sentiment"),
                yaxis=dict(gridcolor=PLOT_THEME['gridcolor'], title="Count"),
                legend=dict(bgcolor='#1b2d3e', font=dict(size=9)),
                height=300
            )
            st.plotly_chart(fig_s, use_container_width=True)

    st.divider()

    st.markdown("#### Topic Performance Heatmap")
    st.caption("Which topics get the most traction on which platforms?")

    heatmap_data = []
    for t_key, t_label in TOPIC_LABELS.items():
        t_df = df[df['Topic'] == t_key]
        heatmap_data.append({
            'Topic':    t_label,
            'Facebook': t_df[t_df['Facebook'] >= 0]['Facebook_score'].mean(),
            'LinkedIn': t_df[t_df['LinkedIn'] >= 0]['LinkedIn_score'].mean(),
            'Google+':  t_df[t_df['GooglePlus'] >= 0]['GooglePlus_score'].mean(),
        })

    heatmap_df = pd.DataFrame(heatmap_data).set_index('Topic')

    heat_fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.tolist(),
        colorscale='YlOrBr',
        text=[[f"{v:.1f}" for v in row] for row in heatmap_df.values],
        texttemplate="%{text}",
        textfont=dict(color='#0d1b2a')
    ))
    heat_fig.update_layout(
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
        font=dict(color=PLOT_THEME['font_color']),
        height=300
    )
    st.plotly_chart(heat_fig, use_container_width=True)

    st.divider()
    st.info(
        "Platform bias is real — the same topic gets amplified differently depending "
        "on the platform's audience and culture. This has direct implications for "
        "how misinformation spreads: platforms that reward emotional content over "
        "credible content become vectors for low-trust viral content."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Credibility Audit
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Credibility Audit")
    st.markdown("Paste any headline you've seen online. This tool analyzes it for credibility signals, manipulation patterns, and tells you which platform it was likely optimized for.")
    st.divider()

    audit_title = st.text_input(
        "Paste a headline to audit",
        placeholder="e.g. You Won't Believe What Scientists Just Discovered!",
        key="audit_title"
    )
    audit_body = st.text_area(
        "Paste article body or summary (optional)",
        placeholder="Paste the article text here for deeper analysis...",
        height=120,
        key="audit_body"
    )

    if st.button("Run Credibility Audit", key="audit_btn"):
        if not audit_title.strip():
            st.error("Please paste a headline to audit.")
        else:
            audit_text   = audit_body.strip() if audit_body.strip() else audit_title
            trust        = compute_trust_score(audit_title, audit_text)
            sentiment    = TextBlob(audit_title).sentiment.polarity
            subjectivity = TextBlob(audit_title).sentiment.subjectivity

            st.divider()

            if trust['score'] >= 70:
                st.success(f"**Verdict: Likely Credible** — Trust score {trust['score']}/100")
            elif trust['score'] >= 45:
                st.warning(f"**Verdict: Mixed Signals** — Trust score {trust['score']}/100")
            else:
                st.error(f"**Verdict: Low Credibility Indicators** — Trust score {trust['score']}/100")

            st.divider()

            st.markdown("### Signal Analysis")
            a1, a2, a3 = st.columns(3)

            with a1:
                st.metric("Credibility Score", f"{trust['score']}/100")
                st.progress(trust['score'] / 100)
            with a2:
                sent_str = "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"
                st.metric("Sentiment", f"{sent_str} ({sentiment:.3f})")
                st.progress(abs(sentiment))
            with a3:
                subj_label = "High" if subjectivity > 0.5 else "Low"
                st.metric("Subjectivity", f"{subj_label} ({subjectivity:.3f})")
                st.progress(subjectivity)

            st.divider()

            st.markdown("### Manipulation Risk Indicators")

            risk_score = 0
            risks      = []
            safe       = []

            caps_ratio = sum(1 for c in audit_title if c.isupper()) / max(len(audit_title), 1)
            if caps_ratio > 0.4:
                risk_score += 25
                risks.append("Excessive capitalization — common in sensational headlines")
            else:
                safe.append("Capitalization appears normal")

            if audit_title.count('!') >= 2:
                risk_score += 20
                risks.append("Multiple exclamation marks — emotional manipulation signal")
            elif audit_title.count('!') == 1:
                risk_score += 10
                risks.append("Exclamation mark detected — mild urgency signal")

            clickbait = ["you won't believe", "shocking", "mind blowing", "what happened next",
                         "they don't want you", "secret", "jaw dropping", "going viral"]
            found_cb  = [p for p in clickbait if p in audit_title.lower()]
            if found_cb:
                risk_score += 30
                risks.append(f"Clickbait language detected: '{found_cb[0]}'")
            else:
                safe.append("No clickbait phrases detected")

            if audit_title.endswith('?'):
                risk_score += 15
                risks.append("Question-bait structure — designed to provoke curiosity without informing")
            else:
                safe.append("Headline structure is declarative, not bait-based")

            if subjectivity > 0.6:
                risk_score += 15
                risks.append("High subjectivity — opinion presented as fact")
            else:
                safe.append("Subjectivity level is within acceptable range")

            risk_score = min(100, risk_score)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Risk signals found:**")
                if risks:
                    for r in risks:
                        st.markdown(f"<span style='color:#e07070'>⚠ {r}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#a8d5a2'>No manipulation signals detected</span>", unsafe_allow_html=True)
            with r2:
                st.markdown("**Credibility signals found:**")
                for s in safe:
                    st.markdown(f"<span style='color:#a8d5a2'>✓ {s}</span>", unsafe_allow_html=True)
                for b in trust['boosts']:
                    st.markdown(f"<span style='color:#a8d5a2'>✓ {b}</span>", unsafe_allow_html=True)

            st.divider()

            st.markdown("### Platform Optimization Detection")
            st.caption("Based on its linguistic signals, this headline appears optimized for:")

            if risk_score > 40 or sentiment < -0.1 or audit_title.count('!') > 0:
                likely_platform = "ⓕ Facebook"
                platform_reason = "Emotional language, urgency signals, and sensational framing are hallmarks of Facebook-optimized content."
            elif ':' in audit_title or any(c.isdigit() for c in audit_title):
                likely_platform = "[in] LinkedIn"
                platform_reason = "Structured format, data references, and professional tone suggest this was written for LinkedIn audiences."
            else:
                likely_platform = "𝐆 Google+"
                platform_reason = "Neutral, informational tone without strong emotional or professional signals suggests Google+ optimization."

            st.info(f"**{likely_platform}** — {platform_reason}")

            st.divider()
            st.caption("Trust by Design · Human-Centered Data Science · Models trained on UCI News Popularity Dataset (93k+ articles)")