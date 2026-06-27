"""
jd_parser.py — TF-IDF based JD understanding and candidate text similarity.

This module builds a TF-IDF representation of the job description, constructs
rich text profiles for candidates, and computes cosine similarity between them.

Design rationale:
- We use TF-IDF (not embeddings) because it runs on CPU with no model downloads,
  fits the ≤5 minute constraint, and is surprisingly effective when the vectorizer
  is configured for the domain (trigrams capture "learning to rank", "vector database",
  "sentence transformers" as single features).
- sublinear_tf=True compresses term frequency via 1+log(tf), preventing a candidate
  who says "python" 20 times from dominating one who says it 3 times.
- We build candidate text from ALL meaningful fields — not just skills — because
  the JD explicitly warns that keyword-in-skills is a trap. Career history
  descriptions reveal what candidates actually DID.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import JD_CORE_TEXT


def build_jd_vectorizer():
    """Build and return a fitted TfidfVectorizer and the JD reference vector.

    The vectorizer is fit on the JD text alone so that all candidate comparisons
    use the JD's vocabulary as the reference frame. This means terms the JD
    doesn't mention get zero weight — exactly the behavior we want for filtering.

    Configuration choices:
    - ngram_range=(1,3): captures single words AND multi-word domain concepts
      like "learning to rank", "sentence transformers", "vector database".
    - max_features=5000: keeps the feature space manageable for 100K candidates
      while retaining all meaningful JD terms (JD only generates ~200 unique
      trigrams, so 5000 is more than enough headroom).
    - sublinear_tf=True: uses 1+log(tf) instead of raw tf, preventing keyword
      stuffing from artificially inflating similarity.
    - lowercase=True + strip_accents='unicode': normalizes text for robust matching.

    Returns:
        tuple: (vectorizer: TfidfVectorizer, jd_vector: sparse matrix of shape (1, n_features))
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        # Standard English stopwords — removes "the", "and", etc. that add noise
        stop_words="english",
    )

    # Fit on JD text. We pass it as a list because sklearn expects an iterable of docs.
    # The vectorizer learns the JD's vocabulary here — only JD terms get features.
    jd_vector = vectorizer.fit_transform([JD_CORE_TEXT])

    return vectorizer, jd_vector


def build_candidate_text(candidate: dict) -> str:
    """Build a rich text representation of a candidate for TF-IDF comparison.

    We concatenate ALL meaningful text fields because the JD warns against
    relying solely on skills keywords. A strong candidate might describe
    "built a recommendation engine" in their career history without listing
    "recommendation systems" as a skill.

    Fields included:
    - headline: often contains role + domain (e.g., "ML Engineer | Search & Ranking")
    - summary: free-text self-description, captures experience narrative
    - career_history descriptions: the richest signal — what they actually DID
    - skill names: explicit claimed skills (but not the only signal!)
    - certification names: e.g., "AWS ML Specialty" signals cloud ML experience

    Args:
        candidate: A single candidate dict from the JSONL data.

    Returns:
        A single lowercase string representing the candidate's full profile text.
        Returns empty string if candidate dict is empty/None.
    """
    if not candidate:
        return ""

    parts = []

    # --- Profile-level text ---
    profile = candidate.get("profile", {}) or {}
    headline = profile.get("headline", "") or ""
    summary = profile.get("summary", "") or ""
    current_title = profile.get("current_title", "") or ""

    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)
    if current_title:
        # Include title separately — it's short but high-signal
        parts.append(current_title)

    # --- Career history descriptions ---
    # This is the MOST valuable text field: it describes what the candidate
    # actually built/shipped, not just what they claim as skills.
    career_history = candidate.get("career_history", []) or []
    for role in career_history:
        if not role:
            continue
        desc = role.get("description", "") or ""
        title = role.get("title", "") or ""
        if desc:
            parts.append(desc)
        if title:
            parts.append(title)

    # --- Skills ---
    # Skill names provide explicit keyword coverage. We include them but
    # don't over-rely on them (the JD explicitly warns this is a trap).
    skills = candidate.get("skills", []) or []
    skill_names = []
    for skill in skills:
        if not skill:
            continue
        name = skill.get("name", "") or ""
        if name:
            skill_names.append(name)
    if skill_names:
        parts.append(" ".join(skill_names))

    # --- Certifications ---
    # Certification names like "AWS ML Specialty" or "TensorFlow Developer"
    # carry domain signal worth capturing in TF-IDF.
    certifications = candidate.get("certifications", []) or []
    cert_names = []
    for cert in certifications:
        if not cert:
            continue
        name = cert.get("name", "") or ""
        if name:
            cert_names.append(name)
    if cert_names:
        parts.append(" ".join(cert_names))

    # Join all parts with spaces; the vectorizer handles tokenization.
    # We do NOT lowercase here — the vectorizer's lowercase=True handles it.
    return " ".join(parts)


def compute_jd_similarity(candidate_text: str, vectorizer, jd_vector) -> float:
    """Compute cosine similarity between a candidate's text and the JD.

    Uses the already-fitted vectorizer to transform the candidate text into
    the same TF-IDF space as the JD, then computes cosine similarity.

    Cosine similarity naturally returns [0, 1] for TF-IDF vectors (all non-negative),
    so no clamping is needed. A score of 0.0 means zero vocabulary overlap with
    the JD; ~0.3+ is typically a strong match given the trigram feature space.

    Args:
        candidate_text: The string output from build_candidate_text().
        vectorizer: The fitted TfidfVectorizer from build_jd_vectorizer().
        jd_vector: The sparse JD vector from build_jd_vectorizer().

    Returns:
        float between 0.0 and 1.0 representing text-level JD alignment.
        Returns 0.0 if candidate_text is empty.
    """
    if not candidate_text or not candidate_text.strip():
        return 0.0

    # Transform uses the fitted vocabulary — terms not in the JD get ignored,
    # which is exactly the behavior we want (we only care about JD-relevant terms).
    candidate_vector = vectorizer.transform([candidate_text])

    # cosine_similarity returns a 2D array; extract the scalar value.
    similarity = cosine_similarity(candidate_vector, jd_vector)[0, 0]

    # Ensure we return a clean float (not numpy scalar) for downstream JSON serialization
    return float(similarity)
