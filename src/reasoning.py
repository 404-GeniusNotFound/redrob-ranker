"""
reasoning.py — Generate specific, honest, varied reasoning for ranked candidates.

Each candidate in the top 100 gets a 1-2 sentence reasoning string that:
- References SPECIFIC facts from their profile (title, company, skills, years)
- Connects to specific JD requirements for Senior AI Engineer
- Acknowledges honest concerns (short tenure, missing skills, etc.)
- Never fabricates information not present in the candidate data
- Varies sentence structure using 5 rotating patterns

The reasoning tone matches rank position:
  Ranks 1-20:  Lead with strengths, briefly note any concern
  Ranks 21-60: Balanced — strengths and limitations
  Ranks 61-100: Acknowledge borderline fit, explain what keeps them in top 100
"""

from typing import Optional

from src.config import (
    WEIGHTS,
    MAX_POINTS,
    MUST_HAVE_SKILL_CONCEPTS,
    NICE_TO_HAVE_SKILL_CONCEPTS,
    CONSULTING_SERVICES_COMPANIES,
    PRODUCT_COMPANIES_KEYWORDS,
    PREFERRED_LOCATIONS,
    WELCOME_LOCATIONS,
    MIN_ACCEPTABLE_TENURE_MONTHS,
)


# ---------------------------------------------------------------------------
# JD requirement labels — human-readable names for what the JD cares about.
# Used to connect a candidate's strength to a specific JD ask.
# ---------------------------------------------------------------------------
_JD_REQUIREMENT_LABELS: dict[str, str] = {
    "embeddings_retrieval": "production embeddings & retrieval experience",
    "vector_databases": "vector database / hybrid search infrastructure",
    "python": "strong Python & ML framework proficiency",
    "evaluation_frameworks": "ranking evaluation framework design (NDCG/MRR/MAP)",
    "ranking_recommendation": "ranking / recommendation system experience",
    "llm_finetuning": "LLM fine-tuning experience (LoRA/QLoRA/PEFT)",
    "learning_to_rank": "learning-to-rank model experience",
    "hrtech_marketplace": "HR-tech / marketplace domain knowledge",
    "distributed_systems": "distributed systems & inference optimization",
    "open_source": "open-source contributions in AI/ML",
}

# Dimension labels for human-readable output
_DIMENSION_LABELS: dict[str, str] = {
    "career_coherence": "career coherence",
    "skill_authenticity": "skill authenticity",
    "jd_semantic_align": "JD alignment",
    "behavioral_avail": "behavioral availability",
    "location_fit": "location fit",
    "education_cred": "education & credibility",
}


# ---------------------------------------------------------------------------
# Internal helpers — extract specific facts from candidate profiles
# ---------------------------------------------------------------------------

def _safe_get_profile(candidate: dict) -> dict:
    """Safely extract the profile sub-dict."""
    return candidate.get("profile", {})


def _get_years(candidate: dict) -> float:
    """Get years of experience, defaulting to 0."""
    return _safe_get_profile(candidate).get("years_of_experience", 0) or 0


def _get_title(candidate: dict) -> str:
    """Get current title, defaulting to empty string."""
    return _safe_get_profile(candidate).get("current_title", "") or ""


def _get_company(candidate: dict) -> str:
    """Get current company, defaulting to empty string."""
    return _safe_get_profile(candidate).get("current_company", "") or ""


def _get_location(candidate: dict) -> str:
    """Get location, defaulting to empty string."""
    return _safe_get_profile(candidate).get("location", "") or ""


def _get_career_companies(candidate: dict) -> list[str]:
    """Get unique company names from career history."""
    career = candidate.get("career_history", [])
    seen = set()
    companies: list[str] = []
    for role in career:
        name = role.get("company", "")
        if name and name not in seen:
            seen.add(name)
            companies.append(name)
    return companies


def _get_top_skills(candidate: dict, n: int = 3) -> list[str]:
    """Get the top N skills sorted by duration descending."""
    skills = candidate.get("skills", [])
    if not skills:
        return []
    sorted_skills = sorted(
        skills,
        key=lambda s: (s.get("duration_months") or 0),
        reverse=True,
    )
    return [s.get("name", "unknown") for s in sorted_skills[:n]]


def _get_avg_tenure_months(candidate: dict) -> Optional[float]:
    """Calculate average tenure across career roles."""
    career = candidate.get("career_history", [])
    if not career:
        return None
    durations = [r.get("duration_months", 0) or 0 for r in career]
    return sum(durations) / len(durations) if durations else None


def _get_best_dimension(scores: dict) -> tuple[str, float]:
    """Find the highest-scoring dimension (normalized by max points).

    Returns (dimension_name, normalized_score) where normalized_score is 0-1.
    """
    best_dim = "career_coherence"
    best_norm = 0.0

    for dim, max_pts in MAX_POINTS.items():
        raw = scores.get(dim, 0) or 0
        norm = raw / max_pts if max_pts > 0 else 0
        if norm > best_norm:
            best_norm = norm
            best_dim = dim

    return best_dim, best_norm


def _get_matched_jd_concepts(candidate: dict) -> list[str]:
    """Find which JD skill concepts the candidate's skills match."""
    skills = candidate.get("skills", [])
    skill_names_lower = {(s.get("name") or "").lower() for s in skills}

    # Also check career descriptions for concept mentions
    career = candidate.get("career_history", [])
    all_desc = " ".join((r.get("description") or "").lower() for r in career)

    matched: list[str] = []

    all_concepts = {**MUST_HAVE_SKILL_CONCEPTS, **NICE_TO_HAVE_SKILL_CONCEPTS}
    for concept, keywords in all_concepts.items():
        for kw in keywords:
            kw_lower = kw.lower()
            # Check if keyword matches any skill name or appears in descriptions
            if any(kw_lower in sn or sn in kw_lower for sn in skill_names_lower if sn):
                matched.append(concept)
                break
            if kw_lower in all_desc:
                matched.append(concept)
                break

    return matched


def _identify_concern(candidate: dict, scores: dict) -> Optional[str]:
    """Identify the most notable concern for this candidate.

    Returns a brief, honest note about a limitation, or None if the
    candidate is strong across all dimensions.
    """
    profile = _safe_get_profile(candidate)
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    concerns: list[str] = []

    # Short tenure / title-chaser risk
    avg_tenure = _get_avg_tenure_months(candidate)
    if avg_tenure is not None and avg_tenure < MIN_ACCEPTABLE_TENURE_MONTHS:
        concerns.append(
            f"avg tenure of {avg_tenure:.0f}mo may indicate job-hopping"
        )

    # Low behavioral availability
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate < 0.3:
        concerns.append(
            f"low recruiter response rate ({response_rate:.0%})"
        )

    # Inactive on platform
    last_active = signals.get("last_active_date", "")
    if last_active and last_active < "2026-01":
        concerns.append("hasn't been active on platform recently")

    # Not open to work
    if not signals.get("open_to_work_flag", True):
        concerns.append("not currently marked open to work")

    # Consulting-only background
    companies = _get_career_companies(candidate)
    if companies:
        all_consulting = all(
            c.lower() in CONSULTING_SERVICES_COMPANIES for c in companies
        )
        if all_consulting and len(companies) > 0:
            concerns.append("career entirely at consulting/services firms")

    # Long notice period
    notice = signals.get("notice_period_days", 30)
    if notice and notice > 60:
        concerns.append(f"{notice}-day notice period")

    # Non-India location
    country = profile.get("country", "")
    if country and country.lower() != "india":
        concerns.append(f"based in {country}")

    # Return the most impactful concern (first one found)
    return concerns[0] if concerns else None


def _is_product_company(company: str) -> bool:
    """Check if a company is a known product company."""
    company_lower = company.lower()
    return any(pc in company_lower for pc in PRODUCT_COMPANIES_KEYWORDS)


# ---------------------------------------------------------------------------
# Pattern generators — each produces a different sentence structure
# ---------------------------------------------------------------------------

def _pattern_a(
    title: str, years: float, company: str,
    strength: str, concern: Optional[str]
) -> str:
    """Pattern A: '[Title] with [X] years at [company]; [strength]. [concern].'"""
    base = f"{title} with {years:.1f} years at {company}; {strength}"
    if concern:
        return f"{base}. However, {concern}."
    return f"{base}."


def _pattern_b(
    dimension_label: str, evidence: str,
    jd_connection: str, concern: Optional[str]
) -> str:
    """Pattern B: 'Strong [dimension] — [evidence]. [JD connection].'"""
    base = f"Strong {dimension_label} profile — {evidence}"
    if jd_connection:
        base += f"; {jd_connection}"
    if concern:
        return f"{base}. Note: {concern}."
    return f"{base}."


def _pattern_c(
    years: float, companies: list[str],
    quality: str, concern: Optional[str]
) -> str:
    """Pattern C: '[X]-year career spanning [companies]; demonstrates [quality].'"""
    company_str = ", ".join(companies[:3])
    base = f"{years:.0f}-year career spanning {company_str}; demonstrates {quality}"
    if concern:
        return f"{base}. Caveat: {concern}."
    return f"{base}."


def _pattern_d(
    skill_domain: str, signal: str, concern: Optional[str]
) -> str:
    """Pattern D: 'Proven [skill/domain] background with [signal]. [limitation].'"""
    base = f"Proven {skill_domain} background with {signal}"
    if concern:
        return f"{base}. Limitation: {concern}."
    return f"{base}."


def _pattern_e(
    achievement: str, behavioral_note: str, concern: Optional[str]
) -> str:
    """Pattern E: 'Career history shows [achievement]; [behavioral signal note].'"""
    base = f"Career history shows {achievement}; {behavioral_note}"
    if concern:
        return f"{base}. {concern.capitalize()}."
    return f"{base}."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reasoning(candidate: dict, rank: int, scores: dict) -> str:
    """Generate a 1-2 sentence reasoning for why this candidate is at this rank.

    Args:
        candidate: The full candidate dict (profile, career_history,
                   skills, redrob_signals, education, etc.)
        rank: The rank position (1-100).
        scores: Dict of dimension scores from the scorer module,
                keyed by dimension name (career_coherence, skill_authenticity, etc.)

    Returns:
        A 1-2 sentence string that is specific, honest, and varied.
        Every fact referenced is taken directly from the candidate's profile.
        Never fabricates skills, companies, or experience.
    """
    # --- Extract candidate facts ---
    title = _get_title(candidate) or "Candidate"
    years = _get_years(candidate)
    company = _get_company(candidate) or "their current company"
    location = _get_location(candidate) or "undisclosed location"
    companies = _get_career_companies(candidate)
    top_skills = _get_top_skills(candidate, n=3)
    best_dim, best_norm = _get_best_dimension(scores)
    matched_concepts = _get_matched_jd_concepts(candidate)
    signals = candidate.get("redrob_signals", {})

    # --- Determine tone based on rank tier ---
    # Ranks 1-20: emphasize strengths, briefly note concern
    # Ranks 21-60: balanced
    # Ranks 61-100: acknowledge borderline, explain what keeps them in
    concern: Optional[str] = None
    if rank > 20:
        # Surface concerns for ranks 21+
        concern = _identify_concern(candidate, scores)
    elif rank > 10:
        # For 11-20, only surface serious concerns
        raw_concern = _identify_concern(candidate, scores)
        if raw_concern and any(kw in (raw_concern or "") for kw in [
            "consulting", "not currently", "hasn't been active"
        ]):
            concern = raw_concern

    # --- Build evidence strings from actual data ---
    dim_label = _DIMENSION_LABELS.get(best_dim, best_dim)

    # Best JD connection
    jd_connection = ""
    if matched_concepts:
        first_match = matched_concepts[0]
        jd_label = _JD_REQUIREMENT_LABELS.get(first_match, first_match)
        jd_connection = f"matches the JD need for {jd_label}"

    # Skill evidence
    skill_str = ", ".join(top_skills[:3]) if top_skills else "broad skills"

    # Behavioral signal note
    response_rate = signals.get("recruiter_response_rate", 0)
    open_to_work = signals.get("open_to_work_flag", False)
    behavioral_note = ""
    if open_to_work and response_rate >= 0.7:
        behavioral_note = "actively seeking roles with strong engagement signals"
    elif open_to_work:
        behavioral_note = "currently open to new opportunities"
    elif response_rate >= 0.6:
        behavioral_note = "good recruiter engagement despite not being marked open to work"
    else:
        behavioral_note = "moderate platform engagement"

    # Product company signal
    has_product_exp = any(_is_product_company(c) for c in companies)
    product_note = "including product-company experience" if has_product_exp else ""

    # --- Borderline reasoning for ranks 61-100 ---
    if rank > 60:
        if concern:
            concern = f"borderline fit — {concern}"
        else:
            concern = "borderline fit for the senior AI engineer profile"

    # --- Select pattern based on rank % 5 for variation ---
    pattern_idx = rank % 5

    if pattern_idx == 0:
        # Pattern A: Title-focused
        strength_parts = []
        if jd_connection:
            strength_parts.append(jd_connection)
        elif top_skills:
            strength_parts.append(f"experienced in {skill_str}")
        else:
            strength_parts.append(f"strong {dim_label}")
        if product_note:
            strength_parts.append(product_note)
        strength = "; ".join(strength_parts)
        return _pattern_a(title, years, company, strength, concern)

    elif pattern_idx == 1:
        # Pattern B: Dimension-focused
        if top_skills:
            evidence = f"{years:.0f} years with depth in {skill_str}"
        else:
            evidence = f"{years:.0f} years of relevant experience"
        return _pattern_b(dim_label, evidence, jd_connection, concern)

    elif pattern_idx == 2:
        # Pattern C: Career-spanning
        quality_parts = []
        if jd_connection:
            quality_parts.append(jd_connection.replace("matches the JD need for ", ""))
        elif dim_label:
            quality_parts.append(f"strong {dim_label}")
        if product_note:
            quality_parts.append(product_note)
        quality = " and ".join(quality_parts) if quality_parts else "relevant experience"
        effective_companies = companies if companies else [company]
        return _pattern_c(years, effective_companies, quality, concern)

    elif pattern_idx == 3:
        # Pattern D: Skill-domain focused
        if top_skills:
            skill_domain = top_skills[0]
        else:
            skill_domain = "technical"

        # Build a specific signal reference
        signal_parts = []
        if years >= 5:
            signal_parts.append(f"{years:.0f}+ years of applied experience")
        else:
            signal_parts.append(f"{years:.1f} years of experience")
        if behavioral_note:
            signal_parts.append(behavioral_note)
        signal = " and ".join(signal_parts[:2])
        return _pattern_d(skill_domain, signal, concern)

    else:
        # Pattern E: Achievement-focused (pattern_idx == 4)
        if companies and years >= 3:
            achievement = (
                f"{years:.0f} years across {len(companies)} "
                f"{'companies' if len(companies) > 1 else 'company'}, "
                f"most recently at {company}"
            )
        elif years >= 1:
            achievement = f"{years:.1f} years at {company} as {title}"
        else:
            achievement = f"early-career profile at {company}"

        return _pattern_e(achievement, behavioral_note, concern)
