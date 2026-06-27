#!/usr/bin/env python3
"""
rank.py — Main entry point for the Redrob Intelligent Candidate Ranker.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Produces a top-100 ranked CSV from a 100K candidate pool in <5 minutes on CPU.
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

from src.config import WEIGHTS, MAX_POINTS
from src.filters import apply_hard_filters
from src.jd_parser import build_jd_vectorizer, build_candidate_text, compute_jd_similarity
from src.features import (
    score_career_coherence,
    score_skill_authenticity,
    score_behavioral_availability,
    score_location_fit,
    score_education_credibility,
)
from src.scorer import compute_candidate_score
from src.honeypot import detect_honeypot
from src.reasoning import generate_reasoning


def load_candidates(filepath: str) -> list[dict]:
    """Load candidates from JSONL or gzipped JSONL file."""
    path = Path(filepath)
    candidates = []

    # Handle .json files (like sample_candidates.json)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        return candidates

    # Handle .jsonl.gz or .jsonl
    if path.suffix.lower() == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")

    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    return candidates


def run_pipeline(candidates: list[dict], verbose: bool = True) -> list[dict]:
    """
    Run the full 5-stage ranking pipeline.

    Returns: List of dicts with keys: candidate_id, rank, score, reasoning
             Sorted by rank (1 = best).
    """
    total = len(candidates)
    t0 = time.time()

    # ── Stage 1: Hard Filters ──────────────────────────────────────────
    if verbose:
        print(f"[Stage 1] Filtering {total:,} candidates...")

    filtered = []
    for c in candidates:
        if apply_hard_filters(c):
            filtered.append(c)

    t1 = time.time()
    if verbose:
        print(f"  → {len(filtered):,} passed filters ({total - len(filtered):,} eliminated) [{t1-t0:.1f}s]")

    # ── Stage 2-3: Feature Engineering + Scoring ───────────────────────
    if verbose:
        print(f"[Stage 2-3] Scoring {len(filtered):,} candidates across 6 dimensions...")

    # Build JD vectorizer once (shared across all candidates)
    vectorizer, jd_vector = build_jd_vectorizer()

    scored_candidates = []
    for i, c in enumerate(filtered):
        # Build candidate text and compute JD similarity
        cand_text = build_candidate_text(c)
        jd_sim = compute_jd_similarity(cand_text, vectorizer, jd_vector)

        # Compute all dimension scores + final weighted score
        scores = compute_candidate_score(c, jd_sim)

        scored_candidates.append({
            "candidate": c,
            "scores": scores,
            "final_score": scores["final_score"],
        })

        if verbose and (i + 1) % 5000 == 0:
            elapsed = time.time() - t1
            print(f"  → Scored {i+1:,}/{len(filtered):,} [{elapsed:.1f}s]")

    t2 = time.time()
    if verbose:
        print(f"  → Scoring complete [{t2-t1:.1f}s]")

    # ── Stage 4: Honeypot Detection ────────────────────────────────────
    if verbose:
        print("[Stage 4] Detecting honeypots...")

    honeypot_count = 0
    for sc in scored_candidates:
        is_hp, reasons = detect_honeypot(sc["candidate"])
        sc["is_honeypot"] = is_hp
        sc["honeypot_reasons"] = reasons
        if is_hp:
            honeypot_count += 1

    t3 = time.time()
    if verbose:
        print(f"  → {honeypot_count} honeypots detected [{t3-t2:.1f}s]")

    # ── Sort and select top 100 (excluding honeypots) ──────────────────
    # Sort all by final_score descending
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Select top 100, preferring non-honeypots
    top_100 = []
    honeypots_in_top = 0
    for sc in scored_candidates:
        if len(top_100) >= 100:
            break
        if sc["is_honeypot"]:
            # Skip honeypots — they'd be tier 0 and hurt our NDCG
            continue
        top_100.append(sc)

    # If we don't have 100 non-honeypot candidates (unlikely), fill with honeypots
    if len(top_100) < 100:
        for sc in scored_candidates:
            if len(top_100) >= 100:
                break
            if sc["is_honeypot"] and sc not in top_100:
                top_100.append(sc)
                honeypots_in_top += 1

    if verbose and honeypots_in_top > 0:
        print(f"  ⚠ Had to include {honeypots_in_top} honeypots to reach 100 candidates")

    # ── Stage 5: Assign Ranks + Generate Reasoning ─────────────────────
    if verbose:
        print("[Stage 5] Generating reasoning for top 100...")

    results = []
    for rank_idx, sc in enumerate(top_100):
        rank = rank_idx + 1
        reasoning = generate_reasoning(sc["candidate"], rank, sc["scores"])
        results.append({
            "candidate_id": sc["candidate"]["candidate_id"],
            "rank": rank,
            "score": round(sc["final_score"], 4),
            "reasoning": reasoning,
        })

    t4 = time.time()
    if verbose:
        print(f"  → Reasoning complete [{t4-t3:.1f}s]")
        print(f"\n{'='*60}")
        print(f"Pipeline complete in {t4-t0:.1f}s")
        print(f"Top candidate: {results[0]['candidate_id']} (score: {results[0]['score']:.4f})")
        print(f"{'='*60}\n")

    return results


def write_submission_csv(results: list[dict], output_path: str):
    """Write the ranked results to a CSV file matching the submission spec.
    
    Format matches the sample: candidate_id,rank,score,reasoning
    Score uses 4 decimal places. Reasoning is unquoted unless it contains commas.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                f"{r['score']:.4f}",
                r["reasoning"],
            ])


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Intelligent Candidate Ranker — "
                    "Rank top 100 candidates for a job description."
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl, candidates.jsonl.gz, or sample_candidates.json"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output CSV path (e.g., ./submission.csv)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.candidates).exists():
        print(f"Error: Input file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    # Load candidates
    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates.\n")

    # Run the pipeline
    results = run_pipeline(candidates, verbose=not args.quiet)

    # Ensure scores are strictly non-increasing (tie-break by candidate_id)
    # This is required by the submission spec
    for i in range(1, len(results)):
        if results[i]["score"] > results[i-1]["score"]:
            results[i]["score"] = results[i-1]["score"]
    # Break ties deterministically by candidate_id ascending
    i = 0
    while i < len(results):
        j = i
        while j < len(results) and results[j]["score"] == results[i]["score"]:
            j += 1
        if j - i > 1:
            # Sort the tied block by candidate_id ascending
            tied = results[i:j]
            tied.sort(key=lambda x: x["candidate_id"])
            for k, t in enumerate(tied):
                t["rank"] = i + k + 1
            results[i:j] = tied
        i = j

    # Write output
    write_submission_csv(results, args.out)
    print(f"Submission written to {args.out}")
    print(f"Run 'python validate_submission.py {args.out}' to validate.")


if __name__ == "__main__":
    main()
