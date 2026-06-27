"""
features.py — Compute the 6 scoring dimensions for each candidate.

This is the heart of the ranking pipeline. Each function maps a raw candidate
dict to a numeric score on a fixed scale. The dimensions are:

  1. Career Coherence       (0-25 pts)  — trajectory, company type, tenure, domain
  2. Skill Authenticity     (0-25 pts)  — depth, coverage, corroboration, assessments
  3. Behavioral Availability(0-15 pts)  — platform engagement & readiness signals
  4. Location Fit           (0-10 pts)  — geographic match to JD preferences
  5. Education Credibility  (0-5 pts)   — field relevance, tier, GitHub, verification

Dimension 6 (JD Semantic Alignment, 0-20 pts) is computed externally via TF-IDF
in the text_similarity module, then plugged in by scorer.py.

Design principles:
  • Every function is a pure function of `candidate: dict` → float.
  • All magic numbers come from src.config.
  • Missing/null fields never crash — they degrade gracefully to low scores.
  • Inner loops are kept minimal so a list-comprehension over 100K dicts is fast.
"""

from datetime import date, datetime
from typing import List, Dict, Optional, Any

from src.config import (
    REFERENCE_DATE,
    CONSULTING_SERVICES_COMPANIES,
    PRODUCT_COMPANIES_KEYWORDS,
    TECH_TITLE_KEYWORDS,
    NON_TECH_TITLES,
    MUST_HAVE_SKILL_CONCEPTS,
    NICE_TO_HAVE_SKILL_CONCEPTS,
    NEGATIVE_SKILL_CONCEPTS,
    BEHAVIORAL_WEIGHTS,
    PREFERRED_LOCATIONS,
    WELCOME_LOCATIONS,
    RELEVANT_EDUCATION_FIELDS,
    IDEAL_AVG_TENURE_MONTHS,
    MIN_ACCEPTABLE_TENURE_MONTHS,
)


# ---------------------------------------------------------------------------
# Internal helpers (module-private)
# ---------------------------------------------------------------------------

def _safe_get(d: dict, *keys, default=None):
    """Drill into nested dicts without KeyError.
    
    _safe_get(candidate, 'redrob_signals', 'open_to_work_flag', default=False)
    """
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
        if current is None:
            return default
    return current


def _lower(val: Any) -> str:
    """Safely lowercase a value; returns '' for None/non-string."""
    if val is None:
        return ""
    return str(val).lower().strip()


def _contains_any(text: str, keywords: list) -> bool:
    """Check if *text* contains any of the *keywords* (all assumed lowercase)."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _parse_date(date_str: str) -> Optional[date]:
    """Parse 'YYYY-MM-DD' strings defensively."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# Pre-compute lowercased sets/lists once at import time for speed.
_CONSULTING_SET = CONSULTING_SERVICES_COMPANIES  # already a set of lowercase strings
_PRODUCT_KW = [kw.lower() for kw in PRODUCT_COMPANIES_KEYWORDS]
_TECH_TITLE_KW = [kw.lower() for kw in TECH_TITLE_KEYWORDS]
_NON_TECH_TITLE_KW = [kw.lower() for kw in NON_TECH_TITLES]

# Flatten all must-have and nice-to-have concepts for quick matching.
_MUST_HAVE_FLAT: Dict[str, List[str]] = {
    concept: [kw.lower() for kw in keywords]
    for concept, keywords in MUST_HAVE_SKILL_CONCEPTS.items()
}
_NICE_TO_HAVE_FLAT: Dict[str, List[str]] = {
    concept: [kw.lower() for kw in keywords]
    for concept, keywords in NICE_TO_HAVE_SKILL_CONCEPTS.items()
}

# Relevant education field keywords (lowercased once)
_EDUCATION_FIELDS_KW = [f.lower() for f in RELEVANT_EDUCATION_FIELDS]

# Tech-industry keywords for career domain relevance scoring
_TECH_INDUSTRY_KEYWORDS = [
    "software", "technology", "tech", "it ", "it services",
    "internet", "computer", "data", "ai", "artificial intelligence",
    "machine learning", "cloud", "saas", "fintech", "edtech",
    "e-commerce", "ecommerce", "analytics", "information",
    "semiconductor", "electronics", "telecom", "cybersecurity",
]

# ML/AI/ranking/retrieval work keywords for role progression scoring
_ML_WORK_KEYWORDS = [
    "machine learning", "ml", "deep learning", "ai", "artificial intelligence",
    "nlp", "natural language", "ranking", "retrieval", "search",
    "recommendation", "embedding", "transformer", "neural network",
    "data science", "model", "training", "inference", "feature engineering",
    "reranking", "re-ranking", "vector", "faiss", "rag",
]


# ============================================================================
# Dimension 1: Career Coherence (0-25 pts)
# ============================================================================

def score_career_coherence(candidate: dict) -> float:
    """Score 0-25: Career trajectory & company-type fit.
    
    Sub-components:
        a) Product vs Services ratio           (0-8 pts)
        b) Role progression quality             (0-7 pts)
        c) Average tenure stability             (0-5 pts)
        d) Career domain relevance (tech/AI)    (0-5 pts)
    """
    career = candidate.get("career_history") or []
    if not career:
        # No career data → minimal score (not zero — could be data issue)
        return 2.0

    # ------------------------------------------------------------------
    # a) Product vs Services ratio (0-8 pts)
    # ------------------------------------------------------------------
    n_roles = len(career)
    n_services = 0
    n_product = 0

    for role in career:
        company_lower = _lower(role.get("company"))
        # Check against the consulting/services set
        if company_lower in _CONSULTING_SET:
            n_services += 1
        # Check against known product-company keywords
        elif any(kw in company_lower for kw in _PRODUCT_KW):
            n_product += 1
        else:
            # Unknown company — check industry as a secondary signal
            industry_lower = _lower(role.get("industry"))
            if industry_lower in ("it services", "consulting", "staffing"):
                n_services += 1
            # Don't auto-classify unknown companies as product; leave neutral

    if n_roles > 0 and n_services == n_roles:
        # ALL companies are consulting/services → JD's explicit disqualifier
        product_score = 0.0
    elif n_product > 0 and n_services == 0:
        # Entirely product companies → full marks
        product_score = 8.0
    elif n_product > n_services:
        # Mostly product
        product_score = 6.0 + 2.0 * (n_product - n_services) / n_roles
        product_score = min(product_score, 8.0)
    elif n_product > 0:
        # Mixed with some product experience
        product_score = 4.0
    else:
        # No known product companies, but not all services either (unknowns)
        product_score = 3.0

    # ------------------------------------------------------------------
    # b) Role progression quality (0-7 pts)
    # ------------------------------------------------------------------
    progression_score = _compute_role_progression(career)

    # ------------------------------------------------------------------
    # c) Average tenure stability (0-5 pts)
    # ------------------------------------------------------------------
    tenures = [
        r.get("duration_months", 0) for r in career
        if r.get("duration_months") is not None and r.get("duration_months") > 0
    ]
    if tenures:
        avg_tenure = sum(tenures) / len(tenures)
        if avg_tenure >= IDEAL_AVG_TENURE_MONTHS:
            tenure_score = 5.0
        elif avg_tenure >= MIN_ACCEPTABLE_TENURE_MONTHS:
            # Linear interpolation between 3 and 5
            tenure_score = 3.0 + 2.0 * (avg_tenure - MIN_ACCEPTABLE_TENURE_MONTHS) / (
                IDEAL_AVG_TENURE_MONTHS - MIN_ACCEPTABLE_TENURE_MONTHS
            )
        else:
            # Title-chaser warning
            tenure_score = 1.0
    else:
        tenure_score = 2.0  # Can't evaluate — give neutral

    # ------------------------------------------------------------------
    # d) Career domain relevance (0-5 pts)
    # ------------------------------------------------------------------
    domain_score = _compute_domain_relevance(career)

    total = product_score + progression_score + tenure_score + domain_score
    return min(total, 25.0)


def _compute_role_progression(career: list) -> float:
    """Evaluate title progression and ML/AI work in descriptions (0-7 pts).
    
    Strategy:
    - Check if titles are in the tech track (vs non-tech)
    - Check if descriptions mention ML/AI/ranking/retrieval work
    - Reward more recent roles being more senior in tech
    """
    if not career:
        return 0.0

    n_roles = len(career)
    tech_role_count = 0
    ml_work_count = 0
    has_recent_senior_tech = False

    for i, role in enumerate(career):
        title_lower = _lower(role.get("title"))
        desc_lower = _lower(role.get("description"))
        is_current = role.get("is_current", False)

        # Is this a tech role?
        is_tech = _contains_any(title_lower, _TECH_TITLE_KW)
        is_non_tech = _contains_any(title_lower, _NON_TECH_TITLE_KW)

        if is_tech and not is_non_tech:
            tech_role_count += 1

            # Check seniority of recent/current tech roles
            if is_current or i == 0:
                seniority_kw = ["senior", "staff", "principal", "lead", "head",
                                "director", "vp", "manager", "architect"]
                if any(kw in title_lower for kw in seniority_kw):
                    has_recent_senior_tech = True

        # Does the description mention ML/AI/ranking/retrieval work?
        if _contains_any(desc_lower, _ML_WORK_KEYWORDS):
            ml_work_count += 1

    # Score tech-track presence (0-3)
    if n_roles > 0:
        tech_ratio = tech_role_count / n_roles
    else:
        tech_ratio = 0.0

    if tech_ratio >= 0.8:
        tech_track_pts = 3.0
    elif tech_ratio >= 0.5:
        tech_track_pts = 2.0
    elif tech_ratio > 0:
        tech_track_pts = 1.0
    else:
        tech_track_pts = 0.0

    # Score ML work mentions (0-2.5)
    if n_roles > 0:
        ml_ratio = ml_work_count / n_roles
    else:
        ml_ratio = 0.0

    if ml_ratio >= 0.6:
        ml_work_pts = 2.5
    elif ml_ratio >= 0.3:
        ml_work_pts = 1.5
    elif ml_ratio > 0:
        ml_work_pts = 0.75
    else:
        ml_work_pts = 0.0

    # Bonus for recent senior tech role (0-1.5)
    senior_bonus = 1.5 if has_recent_senior_tech else 0.0

    return min(tech_track_pts + ml_work_pts + senior_bonus, 7.0)


def _compute_domain_relevance(career: list) -> float:
    """What fraction of career is in tech/software/data/AI industries? (0-5 pts)"""
    if not career:
        return 0.0

    tech_count = 0
    for role in career:
        industry_lower = _lower(role.get("industry"))
        desc_lower = _lower(role.get("description"))

        # Check industry field
        is_tech_industry = any(kw in industry_lower for kw in _TECH_INDUSTRY_KEYWORDS)
        # Fallback: check description for heavy tech signals
        is_tech_desc = any(
            kw in desc_lower
            for kw in ["software", "engineering", "ml", "machine learning",
                        "data pipeline", "api", "microservice", "model",
                        "algorithm", "deploy", "infrastructure"]
        )

        if is_tech_industry or is_tech_desc:
            tech_count += 1

    tech_ratio = tech_count / len(career)

    if tech_ratio >= 0.9:
        return 5.0
    elif tech_ratio >= 0.6:
        return 3.0
    elif tech_ratio >= 0.3:
        return 2.0
    elif tech_ratio > 0:
        return 1.0
    else:
        return 0.0


# ============================================================================
# Dimension 2: Skill Authenticity (0-25 pts)
# ============================================================================

def score_skill_authenticity(candidate: dict) -> float:
    """Score 0-25: Genuine skill depth & coverage.
    
    Sub-components:
        a) Must-have skill concept coverage       (0-12 pts)
        b) Nice-to-have skill concept coverage     (0-5 pts)
        c) Skill–career corroboration              (0-4 pts)
        d) Assessment scores bonus                 (0-4 pts)
    """
    skills = candidate.get("skills") or []
    career = candidate.get("career_history") or []
    signals = candidate.get("redrob_signals") or {}

    # Build a lookup: skill_name_lower → {proficiency, duration_months}
    skill_lookup = {}
    for s in skills:
        name = _lower(s.get("name"))
        if name:
            skill_lookup[name] = {
                "proficiency": _lower(s.get("proficiency")),
                "duration": s.get("duration_months") or 0,
            }

    # Build concatenated career text once for corroboration checks
    career_text = " ".join(
        _lower(r.get("description")) for r in career if r.get("description")
    )

    # ------------------------------------------------------------------
    # a) Must-have skill concept coverage (0-12 pts)
    # ------------------------------------------------------------------
    must_have_score = _score_concept_coverage(
        skill_lookup, _MUST_HAVE_FLAT, max_points=12.0
    )

    # ------------------------------------------------------------------
    # b) Nice-to-have skill concept coverage (0-5 pts)
    # ------------------------------------------------------------------
    nice_to_have_score = _score_concept_coverage(
        skill_lookup, _NICE_TO_HAVE_FLAT, max_points=5.0
    )

    # ------------------------------------------------------------------
    # c) Skill–career corroboration (0-4 pts)
    # ------------------------------------------------------------------
    corroboration_score = _score_skill_corroboration(skill_lookup, career_text)

    # ------------------------------------------------------------------
    # d) Assessment scores bonus (0-4 pts)
    # ------------------------------------------------------------------
    assessment_score = _score_assessments(signals)

    total = must_have_score + nice_to_have_score + corroboration_score + assessment_score
    return min(total, 25.0)


def _proficiency_duration_weight(proficiency: str, duration: int) -> float:
    """Map (proficiency, duration_months) to a 0.0-1.0 credit multiplier.
    
    The JD values *real* depth, not just listing a keyword.
    Expert + long duration → full credit; beginner + short → minimal.
    """
    # Proficiency base
    prof_map = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}
    prof_base = prof_map.get(proficiency, 0.3)

    # Duration modifier — long usage increases trust
    if duration >= 24:
        dur_mod = 1.0
    elif duration >= 12:
        dur_mod = 0.85
    elif duration >= 6:
        dur_mod = 0.65
    elif duration >= 3:
        dur_mod = 0.4
    else:
        dur_mod = 0.2

    return prof_base * dur_mod


def _score_concept_coverage(
    skill_lookup: dict,
    concept_map: Dict[str, List[str]],
    max_points: float,
) -> float:
    """Score how many JD concepts are covered by the candidate's skill list.
    
    For each concept, find the *best* matching skill (highest credit) and
    accumulate points proportionally.
    """
    n_concepts = len(concept_map)
    if n_concepts == 0:
        return 0.0

    pts_per_concept = max_points / n_concepts
    total = 0.0

    for concept, keywords in concept_map.items():
        best_credit = 0.0
        for skill_name, skill_info in skill_lookup.items():
            # Check if this skill matches any keyword for this concept
            if any(kw in skill_name or skill_name in kw for kw in keywords):
                credit = _proficiency_duration_weight(
                    skill_info["proficiency"], skill_info["duration"]
                )
                best_credit = max(best_credit, credit)
        total += best_credit * pts_per_concept

    return total


def _score_skill_corroboration(skill_lookup: dict, career_text: str) -> float:
    """Check if listed skills are backed up by career description text (0-4 pts).
    
    The JD explicitly warns about candidates who list keywords without real work.
    We check whether the candidate's career descriptions mention work related
    to their top skills.
    """
    if not career_text or not skill_lookup:
        return 0.0

    # Focus on skills that are JD-relevant (from must-have + nice-to-have concepts)
    all_jd_keywords = set()
    for keywords in _MUST_HAVE_FLAT.values():
        all_jd_keywords.update(keywords)
    for keywords in _NICE_TO_HAVE_FLAT.values():
        all_jd_keywords.update(keywords)

    # Find candidate skills that match JD concepts
    jd_relevant_skills = []
    for skill_name in skill_lookup:
        if any(kw in skill_name or skill_name in kw for kw in all_jd_keywords):
            jd_relevant_skills.append(skill_name)

    if not jd_relevant_skills:
        # No JD-relevant skills listed → 0 (nothing to corroborate)
        return 0.0

    # For each JD-relevant skill, check if career text mentions related terms
    corroborated = 0
    for skill_name in jd_relevant_skills:
        # Check if the skill name or related terms appear in career text
        if skill_name in career_text:
            corroborated += 1
        else:
            # Try partial matching: e.g., "faiss" skill → "faiss" or "vector"
            # in career text. We use the concept keywords as related terms.
            for concept_kws in list(_MUST_HAVE_FLAT.values()) + list(_NICE_TO_HAVE_FLAT.values()):
                if skill_name in concept_kws or any(
                    kw in skill_name or skill_name in kw for kw in concept_kws
                ):
                    # Check if *any* keyword from this concept appears in career text
                    if any(kw in career_text for kw in concept_kws):
                        corroborated += 1
                        break

    ratio = corroborated / len(jd_relevant_skills)
    return ratio * 4.0


def _score_assessments(signals: dict) -> float:
    """Score based on skill assessment results (0-4 pts).
    
    Redrob's platform assessments are a strong authenticity signal — they show
    the candidate actually demonstrated competence, not just listed a keyword.
    """
    assessments = signals.get("skill_assessment_scores") or {}
    if not assessments:
        return 0.0  # No assessments → neutral (not penalized)

    scores = [v for v in assessments.values() if isinstance(v, (int, float))]
    if not scores:
        return 0.0

    avg_score = sum(scores) / len(scores)

    if avg_score > 70:
        return 4.0
    elif avg_score > 50:
        return 2.0
    elif avg_score > 30:
        return 1.0
    else:
        return 0.5


# ============================================================================
# Dimension 3: Behavioral Availability (0-15 pts)
# ============================================================================

def score_behavioral_availability(candidate: dict) -> float:
    """Score 0-15: Platform engagement & availability signals.
    
    Uses redrob_signals with weights from config.BEHAVIORAL_WEIGHTS.
    A perfect-on-paper candidate who hasn't logged in for 6 months and
    has a 5% response rate is *not actually available* — this dimension
    captures that per the JD's explicit guidance.
    """
    signals = candidate.get("redrob_signals") or {}

    # --- open_to_work ---
    open_to_work = signals.get("open_to_work_flag")
    otw_score = 1.0 if open_to_work else 0.2

    # --- recency (days since last_active_date) ---
    last_active_str = signals.get("last_active_date")
    last_active = _parse_date(last_active_str) if last_active_str else None
    if last_active:
        days_since = (REFERENCE_DATE - last_active).days
        if days_since < 0:
            # last_active is in the future relative to REFERENCE_DATE — very active
            recency_score = 1.0
        elif days_since < 30:
            recency_score = 1.0
        elif days_since < 90:
            recency_score = 0.7
        elif days_since < 180:
            recency_score = 0.4
        else:
            recency_score = 0.1
    else:
        recency_score = 0.3  # Unknown → mildly unfavorable

    # --- response_rate ---
    response_rate = signals.get("recruiter_response_rate")
    rr_score = float(response_rate) if response_rate is not None else 0.3

    # --- response_time ---
    resp_time_hrs = signals.get("avg_response_time_hours")
    if resp_time_hrs is not None and resp_time_hrs >= 0:
        if resp_time_hrs < 24:
            rt_score = 1.0
        elif resp_time_hrs < 72:
            rt_score = 0.7
        elif resp_time_hrs < 168:
            rt_score = 0.4
        else:
            rt_score = 0.1
    else:
        rt_score = 0.3

    # --- notice_period ---
    notice_days = signals.get("notice_period_days")
    if notice_days is not None and notice_days >= 0:
        if notice_days < 30:
            np_score = 1.0
        elif notice_days <= 60:
            np_score = 0.7
        elif notice_days <= 90:
            np_score = 0.4
        else:
            np_score = 0.2
    else:
        np_score = 0.5  # Unknown → neutral

    # --- interview_completion ---
    ic_rate = signals.get("interview_completion_rate")
    ic_score = float(ic_rate) if ic_rate is not None and ic_rate >= 0 else 0.5

    # --- profile_completeness ---
    pc = signals.get("profile_completeness_score")
    pc_score = (float(pc) / 100.0) if pc is not None and pc >= 0 else 0.3

    # --- offer_acceptance ---
    oa_rate = signals.get("offer_acceptance_rate")
    if oa_rate is not None and oa_rate >= 0:
        # -1 means "no data" per schema; non-negative is a real value
        oa_score = float(oa_rate)
    else:
        oa_score = 0.5  # Neutral when no data

    # Combine using configured weights
    weighted_sum = (
        otw_score * BEHAVIORAL_WEIGHTS["open_to_work"]
        + recency_score * BEHAVIORAL_WEIGHTS["recency"]
        + rr_score * BEHAVIORAL_WEIGHTS["response_rate"]
        + rt_score * BEHAVIORAL_WEIGHTS["response_time"]
        + np_score * BEHAVIORAL_WEIGHTS["notice_period"]
        + ic_score * BEHAVIORAL_WEIGHTS["interview_completion"]
        + pc_score * BEHAVIORAL_WEIGHTS["profile_completeness"]
        + oa_score * BEHAVIORAL_WEIGHTS["offer_acceptance"]
    )

    # Scale to 0-15 range
    return weighted_sum * 15.0


# ============================================================================
# Dimension 4: Location Fit (0-10 pts)
# ============================================================================

def score_location_fit(candidate: dict) -> float:
    """Score 0-10: Geographic match to JD's location preferences.
    
    JD priorities: Pune/Noida preferred, tier-1 Indian cities welcome,
    within India + willing to relocate is OK, outside India is unlikely.
    """
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}

    location_raw = _lower(profile.get("location"))
    country_raw = _lower(profile.get("country"))
    willing = signals.get("willing_to_relocate", False)
    work_mode = _lower(signals.get("preferred_work_mode"))

    # Base location score
    if any(city in location_raw for city in PREFERRED_LOCATIONS):
        # Pune or Noida — top match
        base_score = 10.0
    elif any(city in location_raw for city in WELCOME_LOCATIONS):
        # Other tier-1 Indian cities
        base_score = 8.0
    elif country_raw == "india":
        base_score = 6.0 if willing else 4.0
    else:
        # Outside India
        base_score = 2.0 if willing else 0.0

    # Work-mode adjustment: JD is hybrid-flexible
    # 'remote' only is a slight negative — the JD expects occasional in-office
    if work_mode == "remote":
        base_score = max(base_score - 1.0, 0.0)
    # 'hybrid', 'flexible', 'onsite' are all fine — no adjustment

    return min(base_score, 10.0)


# ============================================================================
# Dimension 5: Education Credibility (0-5 pts)
# ============================================================================

def score_education_credibility(candidate: dict) -> float:
    """Score 0-5: Education + credibility / verification signals.
    
    Sub-components:
        a) Education field relevance     (0-1.5 pts)
        b) Institution tier              (0-1.0 pts)
        c) GitHub activity               (0-1.5 pts)
        d) Verification signals          (0-1.0 pts)
    """
    education = candidate.get("education") or []
    signals = candidate.get("redrob_signals") or {}

    # ------------------------------------------------------------------
    # a) Education field relevance (0-1.5 pts) — take best across degrees
    # ------------------------------------------------------------------
    best_field_score = 0.0
    best_tier_score = 0.0

    for edu in education:
        field_lower = _lower(edu.get("field_of_study"))

        # Check for CS / ML / Data Science / Statistics
        cs_core = ["computer science", "computer engineering", "software engineering",
                    "data science", "machine learning", "artificial intelligence",
                    "statistics", "informatics"]
        engineering_other = ["engineering", "electronics", "ece",
                             "electrical engineering", "information technology",
                             "mathematics", "computational"]

        if any(kw in field_lower for kw in cs_core):
            field_pts = 1.5
        elif any(kw in field_lower for kw in engineering_other):
            field_pts = 0.75
        else:
            field_pts = 0.0
        best_field_score = max(best_field_score, field_pts)

        # ------------------------------------------------------------------
        # b) Institution tier (0-1.0 pts) — take best
        # ------------------------------------------------------------------
        tier = _lower(edu.get("tier"))
        tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25}
        tier_pts = tier_map.get(tier, 0.25)
        best_tier_score = max(best_tier_score, tier_pts)

    # ------------------------------------------------------------------
    # c) GitHub activity (0-1.5 pts)
    # ------------------------------------------------------------------
    github = signals.get("github_activity_score")
    if github is not None and github > 0:
        if github > 60:
            github_pts = 1.5
        elif github > 30:
            github_pts = 1.0
        elif github > 10:
            github_pts = 0.5
        else:
            github_pts = 0.25
    else:
        # -1 or missing → 0 (no positive signal, not penalized)
        github_pts = 0.0

    # ------------------------------------------------------------------
    # d) Verification signals (0-1.0 pts)
    # ------------------------------------------------------------------
    verified_email = 1 if signals.get("verified_email") else 0
    verified_phone = 1 if signals.get("verified_phone") else 0
    linkedin = 1 if signals.get("linkedin_connected") else 0
    verification_pts = (verified_email + verified_phone + linkedin) * (1.0 / 3.0)

    total = best_field_score + best_tier_score + github_pts + verification_pts
    return min(total, 5.0)
