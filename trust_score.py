# trust_score.py
import re

CLICKBAIT_PHRASES = [
    "you won't believe", "shocking", "mind blowing", "going viral",
    "breaks the internet", "what happened next", "this is why",
    "the truth about", "secret", "they don't want you to know",
    "number \\d+ will shock you", "jaw dropping"
]

PROFESSIONAL_WORDS = [
    "analysis", "report", "study", "research", "data", "strategy",
    "policy", "growth", "investment", "economy", "market", "technology",
    "innovation", "development", "initiative", "framework", "findings"
]

def compute_trust_score(title, headline):
    score  = 60  # neutral baseline
    flags  = []
    boosts = []

    combined = (title + ' ' + headline).lower()

    # ── Negative signals ─────────────────────────────────────────────────────
    caps_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
    if caps_ratio > 0.4:
        score -= 15
        flags.append("⚠️ Excessive capitalization detected")

    exclamations = title.count('!')
    if exclamations >= 2:
        score -= 10
        flags.append("⚠️ Multiple exclamation marks reduce credibility")
    elif exclamations == 1:
        score -= 5
        flags.append("⚠️ Exclamation mark detected")

    for phrase in CLICKBAIT_PHRASES:
        if re.search(phrase, combined):
            score -= 12
            flags.append(f"⚠️ Clickbait pattern detected: '{phrase}'")
            break

    if title.endswith('?'):
        score -= 8
        flags.append("⚠️ Question-bait headline structure")

    if len(title.split()) < 4:
        score -= 8
        flags.append("⚠️ Title too short — may lack context")

    if len(title.split()) > 18:
        score -= 5
        flags.append("⚠️ Title too long — may lose reader attention")

    # ── Positive signals ─────────────────────────────────────────────────────
    prof_hits = [w for w in PROFESSIONAL_WORDS if w in combined]
    if len(prof_hits) >= 2:
        score += 15
        boosts.append(f"✅ Professional vocabulary detected: {', '.join(prof_hits[:3])}")
    elif len(prof_hits) == 1:
        score += 7
        boosts.append(f"✅ Professional vocabulary: {prof_hits[0]}")

    if ':' in title:
        score += 8
        boosts.append("✅ Structured headline format (colon usage)")

    if '"' in title:
        score += 5
        boosts.append("✅ Contains attribution/quote")

    word_count = len(title.split())
    if 6 <= word_count <= 14:
        score += 10
        boosts.append("✅ Optimal headline length")

    if caps_ratio < 0.15:
        score += 7
        boosts.append("✅ Appropriate capitalization")

    # ── Clamp and label ───────────────────────────────────────────────────────
    score = max(0, min(100, score))

    if score >= 70:
        label = "High"
        color = "green"
    elif score >= 45:
        label = "Medium"
        color = "orange"
    else:
        label = "Low"
        color = "red"

    return {
        'score': score,
        'label': label,
        'color': color,
        'flags': flags,
        'boosts': boosts
    }