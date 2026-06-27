"""
scorer.py — Combine all dimension scores into a final composite score.

This module is the thin orchestration layer that:
  1. Calls each features.py scoring function for a single candidate.
  2. Accepts a pre-computed JD semantic similarity score (from text_similarity).
  3. Normalises each dimension to 0-1 and applies config.WEIGHTS.
  4. Returns a rich dict with per-dimension breakdowns + final score.

Why a separate module?
  • features.py is purely about *computing* individual signals.
  • scorer.py is about *combining* them with the right weights.
  • This separation makes it trivial to swap weighting strategies, add new
    dimensions, or run ablation studies during hackathon tuning.

Performance note:
  All work is pure Python dict manipulation + a handful of function calls.
  On 100K candidates this takes ~30-60 seconds end-to-end (single-threaded),
  well within the 5-minute budget.
"""

from typing import Dict

from src.config import WEIGHTS, MAX_POINTS
from src.features import (
    score_career_coherence,
    score_skill_authenticity,
    score_behavioral_availability,
    score_location_fit,
    score_education_credibility,
)


def compute_candidate_score(candidate: dict, jd_similarity: float) -> Dict[str, float]:
    """Compute all dimension scores and the weighted final score.
    
    Args:
        candidate:     The full candidate dict (profile, career_history,
                       skills, education, redrob_signals, etc.)
        jd_similarity: Pre-computed TF-IDF cosine similarity between the
                       candidate's text and the JD (0.0–1.0 float).
                       Computed externally in text_similarity.py for
                       vectorised efficiency.
    
    Returns:
        dict with keys:
            career_coherence  : float (0-25)
            skill_authenticity: float (0-25)
            jd_semantic_align : float (0-20)
            behavioral_avail  : float (0-15)
            location_fit      : float (0-10)
            education_cred    : float (0-5)
            raw_total         : float (0-100)
            final_score       : float (0.0-1.0, weighted)
    """
    # --- Compute each dimension ---
    career  = score_career_coherence(candidate)
    skill   = score_skill_authenticity(candidate)
    jd_sem  = _scale_jd_similarity(jd_similarity)
    behav   = score_behavioral_availability(candidate)
    loc     = score_location_fit(candidate)
    edu     = score_education_credibility(candidate)

    # --- Raw total (simple sum of all dimensions, 0-100 possible) ---
    raw_total = career + skill + jd_sem + behav + loc + edu

    # --- Weighted final score ---
    # Each dimension is normalised to 0-1 by dividing by its max, then
    # multiplied by its weight.  The result is a 0-1 composite.
    final_score = (
        (career  / MAX_POINTS["career_coherence"])  * WEIGHTS["career_coherence"]
        + (skill / MAX_POINTS["skill_authenticity"]) * WEIGHTS["skill_authenticity"]
        + (jd_sem / MAX_POINTS["jd_semantic_align"]) * WEIGHTS["jd_semantic_align"]
        + (behav / MAX_POINTS["behavioral_avail"])   * WEIGHTS["behavioral_avail"]
        + (loc   / MAX_POINTS["location_fit"])       * WEIGHTS["location_fit"]
        + (edu   / MAX_POINTS["education_cred"])     * WEIGHTS["education_cred"]
    )

    return {
        "career_coherence":  round(career, 4),
        "skill_authenticity": round(skill, 4),
        "jd_semantic_align": round(jd_sem, 4),
        "behavioral_avail":  round(behav, 4),
        "location_fit":      round(loc, 4),
        "education_cred":    round(edu, 4),
        "raw_total":         round(raw_total, 4),
        "final_score":       round(final_score, 6),
    }


def _scale_jd_similarity(sim: float) -> float:
    """Convert a 0-1 cosine similarity into the 0-20 JD alignment score.
    
    TF-IDF cosine similarities are typically low (0.0–0.4 for even good
    matches) because candidate text and JD text use different vocabularies.
    We apply a mild non-linear stretch so that top candidates separate
    better from the pack, while still keeping 0→0 and 1→20.
    
    Mapping:
        raw_sim → stretched = sim^0.6   (compresses the low end, expands the top)
        final   = stretched * 20
    
    This means a raw similarity of 0.10 → 0.25 * 20 = 5.0 pts
    and a raw similarity of 0.30 → 0.49 * 20 = 9.9 pts — reasonable spread.
    """
    # Clamp to [0, 1] defensively
    sim = max(0.0, min(1.0, float(sim)))
    stretched = sim ** 0.6
    return stretched * 20.0


def score_candidates_batch(candidates: list, jd_similarities: list) -> list:
    """Score a batch of candidates efficiently.
    
    Args:
        candidates:     List of candidate dicts.
        jd_similarities: Parallel list of JD similarity floats (same length).
    
    Returns:
        List of score dicts (same order as input).
    
    This is the main entry point for the pipeline — it processes candidates
    sequentially but each call is fast (~0.3ms per candidate).
    """
    assert len(candidates) == len(jd_similarities), (
        f"Mismatch: {len(candidates)} candidates vs {len(jd_similarities)} similarities"
    )

    results = []
    for cand, sim in zip(candidates, jd_similarities):
        results.append(compute_candidate_score(cand, sim))
    return results
