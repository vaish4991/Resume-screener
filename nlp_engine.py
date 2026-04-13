"""
nlp_engine.py
─────────────
Core NLP logic for the AI Resume Screener.
Implements TF-IDF vectorization and cosine similarity from scratch
using only Python standard library + basic math (no sklearn required for core logic).

Optional: if scikit-learn is installed, uses it for more accurate TF-IDF.
"""

import re
import math
from collections import Counter
from datetime import datetime

# ── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','have','has','do','does','will',
    'would','could','should','this','that','these','those','we','our','you',
    'your','it','its','as','by','from','into','about','more','also','not',
    'all','can','may','must','need','well','strong','good','great','excellent',
    'experience','work','working','team','teams','skills','skill','years',
    'year','ability','required','preferred','nice','using','used','use',
    'build','built','develop','developed','knowledge','understanding',
    'familiarity','proven','demonstrated','responsible','including','such',
    'etc','including','various','multiple','several','plus','least','least',
    'across','within','through','between','both','other','any','each',
    'their','they','them','who','what','how','when','where','than','then',
    'new','high','large','small','key','main','major','significant',
}


# ── Tokenizer ────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Lowercase, clean punctuation, remove stopwords, return tokens."""
    text = text.lower()
    # Preserve tech terms like C++, .NET, C#, Node.js
    text = re.sub(r'c\+\+', 'cplusplus', text)
    text = re.sub(r'c#', 'csharp', text)
    text = re.sub(r'\.net', 'dotnet', text)
    text = re.sub(r'node\.js', 'nodejs', text)
    text = re.sub(r'next\.js', 'nextjs', text)
    text = re.sub(r'vue\.js', 'vuejs', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def extract_phrases(text: str, top_n: int = 30) -> list[str]:
    """Extract important unigrams and bigrams ranked by frequency."""
    tokens = tokenize(text)
    phrases = list(tokens)
    for i in range(len(tokens) - 1):
        phrases.append(f"{tokens[i]} {tokens[i+1]}")
    freq = Counter(phrases)
    return [p for p, _ in freq.most_common(top_n)]


# ── TF-IDF (from scratch) ────────────────────────────────────────────────────
def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Term Frequency: normalized by max frequency."""
    counts = Counter(tokens)
    max_count = max(counts.values(), default=1)
    return {term: count / max_count for term, count in counts.items()}


def compute_tfidf(docs: list[str]) -> list[dict[str, float]]:
    """
    Compute TF-IDF vectors for a list of documents.
    Returns list of {term: tfidf_score} dicts.
    """
    N = len(docs)
    tokenized = [tokenize(doc) for doc in docs]

    # Document frequency
    df: dict[str, int] = {}
    for tokens in tokenized:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    vectors = []
    for tokens in tokenized:
        tf = compute_tf(tokens)
        vec = {}
        for term, tf_val in tf.items():
            idf = math.log((N + 1) / (df.get(term, 0) + 1)) + 1
            vec[term] = tf_val * idf
        vectors.append(vec)

    return vectors


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    all_terms = set(vec_a) | set(vec_b)
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in all_terms)
    norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Keyword Extraction ────────────────────────────────────────────────────────
def extract_top_keywords(text: str, top_n: int = 25) -> list[str]:
    """Extract top keywords by TF score (single document)."""
    tokens = tokenize(text)
    tf = compute_tf(tokens)
    return [term for term, _ in sorted(tf.items(), key=lambda x: -x[1])[:top_n]]


def extract_top_tfidf_terms(tfidf_vec: dict, top_n: int = 10) -> list[tuple[str, float]]:
    """Return top N terms from a TF-IDF vector."""
    return sorted(tfidf_vec.items(), key=lambda x: -x[1])[:top_n]


# ── Suggestion Generator ──────────────────────────────────────────────────────
TECH_KEYWORDS = {
    'python','java','javascript','typescript','sql','react','angular','vue',
    'nodejs','aws','azure','gcp','docker','kubernetes','tensorflow','pytorch',
    'scikit','pandas','numpy','spark','kafka','airflow','git','linux','rest',
    'graphql','mongodb','postgresql','mysql','redis','elasticsearch','hadoop',
    'tableau','powerbi','excel','figma','jira','agile','scrum'
}

def generate_suggestions(
    score: int,
    matched: list[str],
    missing: list[str],
    bonus: list[str]
) -> list[str]:
    tips = []

    if missing:
        top3 = ", ".join(f'"{k}"' for k in missing[:3])
        extra = f" (+{len(missing)-3} more)" if len(missing) > 3 else ""
        tips.append(
            f"Add missing keywords to your resume: {top3}{extra}. "
            "Use the exact phrasing from the job description — ATS systems match literal strings."
        )

    if score < 50:
        tips.append(
            f"Your score of {score}% is below the typical ATS threshold of 60%. "
            "Rewrite your professional summary and bullet points to mirror the language in the job description."
        )

    tech_missing = [k for k in missing if k in TECH_KEYWORDS]
    if tech_missing:
        tips.append(
            f"Critical technical skills missing: {', '.join(tech_missing)}. "
            "If you have adjacent experience, describe it using these exact terms. "
            "Otherwise, build 1-2 portfolio projects to demonstrate these skills."
        )

    if len(matched) < 5:
        tips.append(
            f"Only {len(matched)} keywords matched. "
            "Mirror the exact terminology from the JD — for example, use 'REST APIs' not 'web services', "
            "'machine learning' not 'AI models'."
        )

    if len(bonus) > 3:
        tips.append(
            f"You have {len(bonus)} extra skills not in the JD. "
            "Reorder your skills section: put JD-relevant skills first, then additional ones. "
            "ATS systems score earlier content more heavily."
        )

    tips.append(
        "Add a professional summary at the top of your resume with the job title and 3-4 key terms from the JD. "
        "Example: 'Machine Learning Engineer with 4 years of experience in Python, TensorFlow, and MLOps.'"
    )

    if score >= 70:
        tips.append(
            "Strong match! Elevate your resume further by quantifying achievements: "
            "'Reduced model latency by 35%', 'Processed 2M+ records daily', 'Led a team of 5 engineers'. "
            "Numbers make your resume stand out from other keyword-matched candidates."
        )

    return tips[:5]


# ── Report Generator ──────────────────────────────────────────────────────────
def _make_bar(pct: int, width: int = 30) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ── Main Screener Class ───────────────────────────────────────────────────────
class ResumeScreener:
    """Main interface for resume screening analysis."""

    def analyze(self, jd_text: str, resume_text: str) -> dict:
        """
        Run full NLP analysis.
        Returns a dict with: score, matched, missing, bonus, top_tfidf, suggestions
        """
        # TF-IDF vectors
        vectors = compute_tfidf([jd_text, resume_text])
        jd_vec, resume_vec = vectors[0], vectors[1]

        # Similarity score
        similarity = cosine_similarity(jd_vec, resume_vec)
        score = round(similarity * 100)

        # Keyword sets
        jd_keywords = extract_top_keywords(jd_text, top_n=25)
        resume_keywords = set(extract_top_keywords(resume_text, top_n=40))

        matched = [k for k in jd_keywords if k in resume_keywords]
        missing = [k for k in jd_keywords if k not in resume_keywords]
        bonus = [k for k in resume_keywords if k not in jd_keywords][:10]

        # Top TF-IDF terms for JD
        top_tfidf = extract_top_tfidf_terms(jd_vec, top_n=10)

        # Suggestions
        suggestions = generate_suggestions(score, matched, missing, bonus)

        return {
            "score": score,
            "similarity": similarity,
            "matched": matched,
            "missing": missing,
            "bonus": bonus,
            "top_tfidf": top_tfidf,
            "suggestions": suggestions,
        }

    def generate_report(self, result: dict, jd_text: str, resume_text: str) -> str:
        """Generate a plain-text downloadable report."""
        score = result["score"]
        bar = _make_bar(score)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            "=" * 60,
            "  AI RESUME SCREENING REPORT",
            f"  Generated: {now}",
            "=" * 60,
            "",
            f"  MATCH SCORE:  {score}%",
            f"  [{bar}]",
            "",
            "-" * 60,
            "  KEYWORD ANALYSIS",
            "-" * 60,
            "",
            f"  Matched ({len(result['matched'])}):",
            "  " + ", ".join(result["matched"]) if result["matched"] else "  None",
            "",
            f"  Missing ({len(result['missing'])}):",
            "  " + ", ".join(result["missing"]) if result["missing"] else "  None",
            "",
            f"  Bonus skills ({len(result['bonus'])}):",
            "  " + ", ".join(result["bonus"]) if result["bonus"] else "  None",
            "",
            "-" * 60,
            "  TOP TF-IDF TERMS (Job Description)",
            "-" * 60,
            "",
        ]

        max_val = result["top_tfidf"][0][1] if result["top_tfidf"] else 1
        for term, val in result["top_tfidf"]:
            pct = int((val / max_val) * 100)
            mini_bar = _make_bar(pct, width=20)
            matched_marker = " ✓" if term in result["matched"] else ""
            lines.append(f"  {term:<18} [{mini_bar}] {val:.3f}{matched_marker}")

        lines += [
            "",
            "-" * 60,
            "  IMPROVEMENT SUGGESTIONS",
            "-" * 60,
            "",
        ]
        for i, tip in enumerate(result["suggestions"], 1):
            lines.append(f"  {i}. {tip}")
            lines.append("")

        lines += [
            "=" * 60,
            "  END OF REPORT",
            "=" * 60,
        ]

        return "\n".join(lines)
