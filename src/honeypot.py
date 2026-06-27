"""
honeypot.py — Detect honeypot candidates with subtly impossible profiles.

The dataset contains ~80 honeypot candidates planted with deliberately
inconsistent profiles (e.g., 8 years claimed experience at a 3-year-old
company, "expert" in 10 skills with 0 months of use, Marketing Manager
whose descriptions are all about ML pipelines).

Any candidate flagged as a honeypot is forced to relevance tier 0.
Submissions with >10% honeypots in the top 100 are disqualified.

Detection strategy: run 6 independent heuristic checks. A candidate is
flagged as a honeypot if 2 or more checks trigger — a single anomaly
could be a data entry error, but *two* correlated impossibilities indicate
a planted profile.
"""

from datetime import date, datetime
from typing import Optional

from src.config import (
    HONEYPOT_EXPERIENCE_DATE_MISMATCH_MONTHS,
    HONEYPOT_MAX_EXPERT_SKILLS_ZERO_DURATION,
    NON_TECH_TITLES,
    REFERENCE_DATE,
    TECH_TITLE_KEYWORDS,
    MUST_HAVE_SKILL_CONCEPTS,
    NICE_TO_HAVE_SKILL_CONCEPTS,
)

# ---------------------------------------------------------------------------
# Internal keyword sets (precomputed once at import time for speed)
# ---------------------------------------------------------------------------

# All AI/ML-related surface forms from config — used for keyword-stuffing check
_AI_SKILL_KEYWORDS: set[str] = set()
for _concept_list in MUST_HAVE_SKILL_CONCEPTS.values():
    _AI_SKILL_KEYWORDS.update(kw.lower() for kw in _concept_list)
for _concept_list in NICE_TO_HAVE_SKILL_CONCEPTS.values():
    _AI_SKILL_KEYWORDS.update(kw.lower() for kw in _concept_list)

# Additional AI/ML keywords that candidates might list as skills
_AI_SKILL_KEYWORDS.update({
    "machine learning", "deep learning", "neural network", "neural networks",
    "nlp", "natural language processing", "computer vision", "transformers",
    "bert", "gpt", "llm", "large language model", "generative ai",
    "reinforcement learning", "tensorflow", "pytorch", "keras", "scikit-learn",
    "data science", "ai", "artificial intelligence",
})

# Tech-related keywords that would appear in descriptions of actual engineers
_TECH_DESCRIPTION_KEYWORDS: set[str] = {
    "pipeline", "api", "deploy", "model", "algorithm", "infrastructure",
    "database", "microservice", "backend", "frontend", "cloud", "aws",
    "gcp", "azure", "docker", "kubernetes", "ml", "training", "inference",
    "embedding", "vector", "retrieval", "ranking", "recommendation",
    "production", "latency", "throughput", "feature engineering",
    "data pipeline", "etl", "spark", "airflow", "kafka",
}

# Non-tech description keywords — domains that non-tech titles work in
_NON_TECH_DESCRIPTION_KEYWORDS: set[str] = {
    "accounting", "ledger", "gaap", "tax", "audit", "compliance",
    "marketing", "branding", "seo", "campaign", "content",
    "hr", "recruiting", "onboarding", "payroll", "benefits",
    "sales", "quota", "crm", "territory", "pipeline",  # sales pipeline != data pipeline
    "logistics", "warehouse", "fulfillment", "supply chain",
    "design", "brand", "packaging", "creative", "photoshop", "figma",
    "support", "tickets", "escalation", "customer",
}


# ---------------------------------------------------------------------------
# Helper: safe date parsing
# ---------------------------------------------------------------------------
def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse a date string (YYYY-MM-DD) safely, returning None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _is_non_tech_title(title: str) -> bool:
    """Check if a title matches known non-tech patterns."""
    title_lower = title.lower().strip()
    return any(nt in title_lower for nt in NON_TECH_TITLES)


def _is_tech_title(title: str) -> bool:
    """Check if a title matches known tech/engineering patterns."""
    title_lower = title.lower().strip()
    return any(tk in title_lower for tk in TECH_TITLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Individual honeypot checks
# ---------------------------------------------------------------------------

def _check_experience_date_mismatch(candidate: dict) -> Optional[str]:
    """Check 1: claimed years_of_experience vs actual career timeline.

    If someone claims 10 years but their career_history only sums to 4 years,
    that's a planted anomaly. We also check whether the earliest career start
    date is too recent for the claimed experience.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    claimed_years = profile.get("years_of_experience", 0) or 0

    if not career or claimed_years <= 0:
        return None

    # Sum actual career duration from duration_months fields
    total_career_months = sum(
        (role.get("duration_months") or 0) for role in career
    )

    claimed_months = claimed_years * 12
    gap = claimed_months - total_career_months

    # Flag if claimed experience exceeds documented career by >2 years
    if gap > HONEYPOT_EXPERIENCE_DATE_MISMATCH_MONTHS:
        return (
            f"Experience-date mismatch: claims {claimed_years:.1f}yr "
            f"({claimed_months:.0f}mo) but career history totals "
            f"{total_career_months}mo (gap={gap:.0f}mo)"
        )

    # Also check: is the earliest start_date too recent?
    earliest_start = None
    for role in career:
        start = _parse_date(role.get("start_date"))
        if start and (earliest_start is None or start < earliest_start):
            earliest_start = start

    if earliest_start:
        # How many months from earliest_start to reference date?
        months_span = (
            (REFERENCE_DATE.year - earliest_start.year) * 12
            + (REFERENCE_DATE.month - earliest_start.month)
        )
        # If claimed experience is much more than the calendar span, flag
        if claimed_months - months_span > HONEYPOT_EXPERIENCE_DATE_MISMATCH_MONTHS:
            return (
                f"Timeline impossible: claims {claimed_years:.1f}yr but "
                f"earliest career start is {earliest_start} "
                f"(only {months_span}mo ago)"
            )

    return None


def _check_skill_inflation(candidate: dict) -> Optional[str]:
    """Check 2: impossible skill proficiency claims.

    Expert proficiency requires years of practice. If someone claims 'expert'
    in a skill but has <=3 months of duration (or 0 endorsements AND 0
    duration), that's synthetic. Flag if >3 such skills exist.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return None

    inflated_skills: list[str] = []

    for skill in skills:
        prof = (skill.get("proficiency") or "").lower()
        duration = skill.get("duration_months") or 0
        endorsements = skill.get("endorsements") or 0
        name = skill.get("name", "unknown")

        if prof == "expert":
            # Expert with trivial duration — impossible
            if duration <= 3:
                inflated_skills.append(f"{name}(expert,{duration}mo)")
            # Expert with zero endorsements AND very low duration — also suspicious
            elif endorsements == 0 and duration == 0:
                inflated_skills.append(f"{name}(expert,0endorse,0mo)")

    threshold = HONEYPOT_MAX_EXPERT_SKILLS_ZERO_DURATION
    if len(inflated_skills) > threshold:
        return (
            f"Skill inflation: {len(inflated_skills)} 'expert' skills with "
            f"<=3mo duration (threshold={threshold}): "
            f"{', '.join(inflated_skills[:5])}"
        )

    return None


def _check_title_description_mismatch(candidate: dict) -> Optional[str]:
    """Check 3: non-tech title but tech/AI career descriptions.

    A Marketing Manager whose career descriptions talk about building ML
    pipelines and deploying models is a planted profile. Conversely, an
    engineer whose descriptions are all about accounting is also suspicious.

    We use keyword matching: count tech keywords vs non-tech keywords in
    descriptions, then compare against the title's domain.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    current_title = profile.get("current_title", "")

    if not current_title or not career:
        return None

    # Collect all description text
    all_desc_text = " ".join(
        (role.get("description") or "").lower() for role in career
    )
    if not all_desc_text.strip():
        return None

    # Count tech vs non-tech keyword hits in descriptions
    tech_hits = sum(1 for kw in _TECH_DESCRIPTION_KEYWORDS if kw in all_desc_text)
    non_tech_hits = sum(1 for kw in _NON_TECH_DESCRIPTION_KEYWORDS if kw in all_desc_text)

    title_is_non_tech = _is_non_tech_title(current_title)
    title_is_tech = _is_tech_title(current_title)

    # Non-tech title but descriptions are overwhelmingly technical
    if title_is_non_tech and tech_hits >= 5 and tech_hits > non_tech_hits * 2:
        return (
            f"Title-description mismatch: title '{current_title}' is non-tech "
            f"but descriptions contain {tech_hits} tech keywords vs "
            f"{non_tech_hits} non-tech keywords"
        )

    # Tech title but descriptions are overwhelmingly non-technical
    # (less common but still a honeypot vector)
    if title_is_tech and non_tech_hits >= 5 and non_tech_hits > tech_hits * 2:
        return (
            f"Title-description mismatch: title '{current_title}' is tech "
            f"but descriptions contain {non_tech_hits} non-tech keywords vs "
            f"{tech_hits} tech keywords"
        )

    return None


def _check_impossible_timeline(candidate: dict) -> Optional[str]:
    """Check 4: career role duration vs actual date span.

    If a role says duration_months=60 but start_date to end_date is only
    12 months, that's physically impossible. We allow ±6 months tolerance
    for rounding and edge effects.
    """
    career = candidate.get("career_history", [])
    if not career:
        return None

    TOLERANCE_MONTHS = 6
    violations: list[str] = []

    for role in career:
        claimed_dur = role.get("duration_months")
        if not claimed_dur:
            continue

        start = _parse_date(role.get("start_date"))
        # For current roles, use reference date as effective end
        end = _parse_date(role.get("end_date"))
        if role.get("is_current") and not end:
            end = REFERENCE_DATE

        if not start or not end:
            continue

        # Calculate actual months between dates
        actual_months = (
            (end.year - start.year) * 12 + (end.month - start.month)
        )

        diff = abs(claimed_dur - actual_months)
        if diff > TOLERANCE_MONTHS:
            company = role.get("company", "unknown")
            violations.append(
                f"{company}: claims {claimed_dur}mo but dates span "
                f"{actual_months}mo (diff={diff}mo)"
            )

    if violations:
        return f"Impossible timeline in {len(violations)} role(s): {violations[0]}"

    return None


def _check_keyword_stuffing(candidate: dict) -> Optional[str]:
    """Check 5: AI keyword stuffing by a non-tech profile.

    If someone has >15 skills AND most are AI-related AND their current_title
    is non-tech AND their descriptions don't actually mention any of their
    listed skills — that's a stuffed profile.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    current_title = profile.get("current_title", "")

    if not skills or len(skills) <= 15:
        return None

    if not _is_non_tech_title(current_title):
        return None

    # Count how many skills are AI-related
    ai_skill_count = 0
    skill_names_lower: list[str] = []
    for skill in skills:
        name = (skill.get("name") or "").lower()
        skill_names_lower.append(name)
        if any(ai_kw in name or name in ai_kw for ai_kw in _AI_SKILL_KEYWORDS):
            ai_skill_count += 1

    # Most skills should be AI-related to flag
    if ai_skill_count < len(skills) * 0.5:
        return None

    # Check if descriptions mention ANY of the listed skills
    all_desc_text = " ".join(
        (role.get("description") or "").lower() for role in career
    )
    skills_mentioned_in_desc = sum(
        1 for name in skill_names_lower
        if name and name in all_desc_text
    )

    # If very few of their listed skills appear in descriptions → stuffing
    mention_ratio = skills_mentioned_in_desc / max(len(skills), 1)
    if mention_ratio < 0.15:
        return (
            f"Keyword stuffing: {len(skills)} skills ({ai_skill_count} AI-related) "
            f"but only {skills_mentioned_in_desc} mentioned in career descriptions. "
            f"Title: '{current_title}'"
        )

    return None


def _check_abnormal_signal_combination(candidate: dict) -> Optional[str]:
    """Check 6: too-perfect behavioral signals on a suspicious profile.

    A profile with ALL of response_rate=1.0, interview_completion=1.0,
    and profile_completeness=100 is statistically implausible for real users.
    When combined with other suspicious signals (non-tech title, inflated
    skills, etc.), it strongly indicates a planted profile.
    """
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    response_rate = signals.get("recruiter_response_rate", 0)
    interview_rate = signals.get("interview_completion_rate", 0)
    completeness = signals.get("profile_completeness_score", 0)
    offer_rate = signals.get("offer_acceptance_rate", -1)

    # Count how many signals are "perfect"
    perfect_count = 0
    if response_rate >= 0.99:
        perfect_count += 1
    if interview_rate >= 0.99:
        perfect_count += 1
    if completeness >= 99:
        perfect_count += 1
    if offer_rate >= 0.99:
        perfect_count += 1

    # Need at least 3 perfect signals to flag
    if perfect_count < 3:
        return None

    # Only flag if the profile itself is also suspicious (non-tech title
    # or experience anomaly) — perfect signals alone aren't honeypot evidence
    current_title = profile.get("current_title", "")
    is_suspicious = _is_non_tech_title(current_title)

    # Or check for skill inflation as additional suspicion
    skills = candidate.get("skills", [])
    expert_low_dur = sum(
        1 for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
        and (s.get("duration_months") or 0) <= 3
    )
    if expert_low_dur > 2:
        is_suspicious = True

    if is_suspicious:
        return (
            f"Abnormal signals: {perfect_count} perfect behavioral scores "
            f"(response={response_rate:.2f}, interview={interview_rate:.2f}, "
            f"completeness={completeness:.1f}) on a suspicious profile "
            f"(title='{current_title}')"
        )

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """Detect if a candidate is likely a honeypot with an impossible profile.

    Runs 6 independent heuristic checks. A candidate is flagged as a
    honeypot if 2 or more checks trigger — a single anomaly could be a
    data entry error, but two correlated impossibilities indicate a
    planted profile.

    Args:
        candidate: Full candidate dict with profile, career_history,
                   skills, redrob_signals, etc.

    Returns:
        (is_honeypot, reasons): bool flag and list of human-readable
        reason strings explaining which checks triggered.
    """
    # Guard against empty / malformed input
    if not candidate or not isinstance(candidate, dict):
        return False, []

    # Run all checks — each returns a reason string or None
    checks = [
        _check_experience_date_mismatch,
        _check_skill_inflation,
        _check_title_description_mismatch,
        _check_impossible_timeline,
        _check_keyword_stuffing,
        _check_abnormal_signal_combination,
    ]

    reasons: list[str] = []
    for check_fn in checks:
        try:
            result = check_fn(candidate)
            if result:
                reasons.append(result)
        except Exception:
            # Defensive: never let a single check crash the pipeline.
            # A broken check simply doesn't contribute to the honeypot score.
            pass

    # Threshold: 2+ triggered checks → honeypot
    is_honeypot = len(reasons) >= 2
    return is_honeypot, reasons
