"""
filters.py — Stage 1 hard filters for rapid candidate elimination.

These filters run BEFORE any scoring to eliminate clearly irrelevant candidates
(~90% of the 100K pool). They must be:
1. FAST — simple string/numeric checks, no model inference
2. LOOSE — the JD explicitly warns that keyword filtering is a trap; we'd rather
   let a borderline candidate through to scoring than miss a "plain-language Tier 5"
3. DEFENSIVE — handle missing/null fields gracefully (real data is messy)

Design philosophy:
- Experience filter: uses a generous band (2-20 years) from config, not the JD's
  stated 5-9, because the JD says "we'll seriously consider candidates outside
  the band if other signals are strong."
- Tech relevance: uses 4 independent heuristics with OR logic — a candidate passes
  if ANY ONE of them fires. This catches the "hidden gem" who describes building
  a recommendation system in their career history but doesn't list "ML" as a skill.
- Location is NOT a hard filter — the JD says "case-by-case" for outside India,
  so location becomes a scoring dimension instead.
"""

import re

from src.config import (
    MIN_EXPERIENCE_YEARS,
    MAX_EXPERIENCE_YEARS,
    TECH_TITLE_KEYWORDS,
    MUST_HAVE_SKILL_CONCEPTS,
    NICE_TO_HAVE_SKILL_CONCEPTS,
)


# Pre-compile a regex pattern for tech-relevance checks against free text.
# These terms indicate ML/AI/data/engineering work in summaries and descriptions.
# Kept broad on purpose — we want to catch "built a search system" not just "ML engineer".
_TECH_TEXT_KEYWORDS = [
    "machine learning", "deep learning", "artificial intelligence",
    "ml", "ai", "data science", "data engineering", "data pipeline",
    "nlp", "natural language processing", "information retrieval",
    "embeddings", "neural network", "recommendation", "ranking",
    "search system", "retrieval", "classification", "regression",
    "clustering", "feature engineering", "model training",
    "model serving", "inference", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "python",
    "vector database", "elasticsearch", "llm", "large language model",
    "fine-tuning", "fine tuning", "transformer", "bert", "gpt",
    "rag", "retrieval augmented", "a/b test", "evaluation",
    "software engineer", "backend", "full stack", "api",
    "microservices", "distributed systems", "cloud",
]

# Build a single compiled regex for efficiency — OR-joined, word-boundary aware
# where possible. We use \b for most terms but handle short terms (ml, ai) carefully.
_TECH_TEXT_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in _TECH_TEXT_KEYWORDS),
    re.IGNORECASE,
)

# Flatten all skill concept surface forms into a single set for fast O(1) lookups.
# Combines both MUST_HAVE and NICE_TO_HAVE because the filter stage just needs
# to know "is this person in the tech/ML universe at all?"
_ALL_SKILL_SURFACE_FORMS = set()
for _concept_group in (MUST_HAVE_SKILL_CONCEPTS, NICE_TO_HAVE_SKILL_CONCEPTS):
    for _forms in _concept_group.values():
        for _form in _forms:
            _ALL_SKILL_SURFACE_FORMS.add(_form.lower())


def _safe_get_text(obj: dict, *keys) -> str:
    """Safely extract and lowercase a nested text field.

    Traverses nested dicts using the provided keys. Returns empty string
    if any key is missing or the value is None/non-string.

    Args:
        obj: The dict to extract from.
        *keys: Sequence of keys to traverse (e.g., "profile", "headline").

    Returns:
        Lowercased string value, or "" if not found/null.
    """
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
        if current is None:
            return ""
    return str(current).lower() if current else ""


def has_tech_relevance(candidate: dict) -> bool:
    """Check if candidate has ANY tech/engineering relevance.

    This filter is deliberately LOOSE. The JD explicitly warns:
    'The "right answer" is not "find candidates whose skills section contains
    the most AI keywords." That's a trap.'

    A candidate passes if ANY of these 4 heuristics fire:

    1. TITLE CHECK: Current or any past title contains tech keywords from config.
       Catches: "ML Engineer", "Data Scientist", "Software Developer", etc.

    2. SUMMARY/HEADLINE CHECK: Free-text mentions of ML/AI/data/engineering work.
       Catches: candidates who describe their work in plain language.

    3. SKILL OVERLAP CHECK: ≥3 skills overlap with MUST_HAVE or NICE_TO_HAVE concepts.
       Threshold is 3 (not 1) to avoid false positives from random skill noise —
       e.g., a marketing person with "Python" listed shouldn't pass on that alone.

    4. CAREER DESCRIPTION CHECK: Any career_history description mentions tech/ML work.
       This is the key heuristic for catching "plain-language Tier 5s" whose career
       descriptions show they built ranking/recommendation/search systems without
       using buzzwords in their skills list.

    Args:
        candidate: A single candidate dict from the JSONL data.

    Returns:
        True if the candidate should pass through to scoring.
    """
    if not candidate:
        return False

    # --- Heuristic 1: Title-based check ---
    # Check current title AND all historical titles.
    titles_to_check = []
    current_title = _safe_get_text(candidate, "profile", "current_title")
    if current_title:
        titles_to_check.append(current_title)

    career_history = candidate.get("career_history", []) or []
    for role in career_history:
        if not role:
            continue
        title = (role.get("title", "") or "").lower()
        if title:
            titles_to_check.append(title)

    for title in titles_to_check:
        for keyword in TECH_TITLE_KEYWORDS:
            if keyword in title:
                return True

    # --- Heuristic 2: Summary/headline mentions tech work ---
    headline = _safe_get_text(candidate, "profile", "headline")
    summary = _safe_get_text(candidate, "profile", "summary")
    profile_text = f"{headline} {summary}"

    if _TECH_TEXT_PATTERN.search(profile_text):
        return True

    # --- Heuristic 3: Skill overlap with JD concepts ---
    # We need ≥3 matching skills to avoid false positives from noise.
    # A single "Python" skill on a marketing manager isn't enough.
    skills = candidate.get("skills", []) or []
    matching_skill_count = 0
    for skill in skills:
        if not skill:
            continue
        skill_name = (skill.get("name", "") or "").lower()
        if not skill_name:
            continue
        # Check if this skill name matches any surface form OR is a substring match.
        # Substring matching catches "PyTorch" matching "pytorch", "Sentence Transformers"
        # matching "sentence transformers", etc.
        if skill_name in _ALL_SKILL_SURFACE_FORMS:
            matching_skill_count += 1
        else:
            # Also check if any surface form is contained within the skill name
            # (handles "Advanced Python Programming" matching "python")
            for form in _ALL_SKILL_SURFACE_FORMS:
                if form in skill_name or skill_name in form:
                    matching_skill_count += 1
                    break  # Count each skill at most once

        if matching_skill_count >= 3:
            return True

    # --- Heuristic 4: Career description mentions tech/ML work ---
    # This catches "plain-language Tier 5s" — people who describe building
    # recommendation systems, search infrastructure, etc. without buzzwords.
    for role in career_history:
        if not role:
            continue
        desc = (role.get("description", "") or "").lower()
        if desc and _TECH_TEXT_PATTERN.search(desc):
            return True

    # None of the 4 heuristics fired — this candidate is likely non-technical
    return False


def passes_experience_filter(candidate: dict) -> bool:
    """Check if candidate's experience falls within the generous band.

    Uses MIN_EXPERIENCE_YEARS (2) and MAX_EXPERIENCE_YEARS (20) from config.
    This is deliberately wider than the JD's stated 5-9 years because:
    - The JD says "we'll seriously consider candidates outside the band"
    - A 4-year prodigy or a 12-year veteran shouldn't be hard-filtered out
    - The exact experience range becomes a scoring factor, not a binary gate

    Edge cases handled:
    - Missing years_of_experience: defaults to True (don't filter out unknowns,
      let scoring handle them — better a false positive than a missed gem)
    - years_of_experience = 0 or negative: treated as missing data

    Args:
        candidate: A single candidate dict from the JSONL data.

    Returns:
        True if candidate's experience is within the acceptable band or unknown.
    """
    if not candidate:
        return False

    profile = candidate.get("profile", {}) or {}
    yoe = profile.get("years_of_experience")

    # Missing or null experience — let them through, scoring will handle it.
    # Rationale: some profiles may not report experience; excluding them risks
    # losing candidates who simply didn't fill in this field.
    if yoe is None:
        return True

    try:
        yoe = float(yoe)
    except (ValueError, TypeError):
        # Non-numeric experience value — treat as missing, let through
        return True

    # Zero or negative experience is likely bad data, let scoring handle
    if yoe <= 0:
        return True

    return MIN_EXPERIENCE_YEARS <= yoe <= MAX_EXPERIENCE_YEARS


def apply_hard_filters(candidate: dict) -> bool:
    """Apply all hard filters. Returns True if candidate should proceed to scoring.

    This is the single entry point for Stage 1 filtering. It chains the filters
    in order of computational cost (cheapest first for early exit):

    1. Experience band check — simple numeric comparison (fastest)
    2. Tech relevance check — string matching against titles/skills/descriptions

    Filters NOT applied here:
    - Location: The JD says "case-by-case" for outside India, so location is a
      SCORING dimension, not a hard filter. A brilliant candidate in Singapore
      should still appear in results, just scored lower on location fit.
    - Salary: Not mentioned as a hard filter in the JD.
    - Notice period: JD says "30+ day candidates still in scope" — it's a scoring
      factor, not a gate.

    Args:
        candidate: A single candidate dict from the JSONL data.

    Returns:
        True if candidate passes all hard filters and should proceed to scoring.
    """
    if not candidate:
        return False

    # Filter 1: Experience band (cheap — one numeric comparison)
    if not passes_experience_filter(candidate):
        return False

    # Filter 2: Tech relevance (slightly more expensive — string matching)
    if not has_tech_relevance(candidate):
        return False

    return True
