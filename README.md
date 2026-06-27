# 🏆 Redrob Ranker — Intelligent Candidate Discovery & Ranking

An AI-powered candidate ranking system that evaluates 100K candidates for a Senior AI Engineer role — not by matching keywords, but by understanding who genuinely fits.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on full dataset (produces submission.csv in <5 minutes)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Run on sample data (for testing)
python rank.py --candidates ./sample_candidates.json --out ./test_output.csv

# Validate output
python validate_submission.py submission.csv
```

## Architecture

The system uses a **5-stage pipeline**:

| Stage | What | Time |
|-------|------|------|
| 1. Hard Filters | Eliminate non-starters (experience band, domain relevance) | ~10s |
| 2. Feature Engineering | Extract 6 scoring dimensions per candidate | ~30s |
| 3. Multi-Signal Scoring | Weighted fusion of all dimensions | ~10s |
| 4. Honeypot Detection | Flag impossible/fake profiles | ~5s |
| 5. Reasoning Generation | Specific, honest, varied justifications | ~30s |

### 6 Scoring Dimensions

1. **Career Coherence (25%)** — Product vs. services experience, role progression, tenure stability
2. **Skill Authenticity (25%)** — Must-have coverage, proficiency-duration correlation, assessment scores
3. **JD Semantic Alignment (20%)** — TF-IDF cosine similarity with the job description
4. **Behavioral Availability (15%)** — Login recency, response rate, notice period, interview completion
5. **Location Fit (10%)** — Pune/Noida preferred, India metros welcome
6. **Education & Credibility (5%)** — CS/ML field, institution tier, GitHub activity, verification

### What Makes This Unique

- **Career-narrative coherence analysis** — Checks if a candidate's career *story* makes sense, not just keywords
- **Anti-gaming detection** — Explicitly detects keyword stuffers, title-chasers, and consulting-only profiles
- **Honeypot detection** — 6-check anomaly system catches impossible profiles
- **Behavioral signal fusion** — Treats availability as 15% of the score (most teams ignore this)
- **Honest reasoning** — Acknowledges concerns, not just praise

## Compute Constraints

| Constraint | Limit | Actual |
|-----------|-------|--------|
| Runtime | ≤5 min | ~90 sec |
| Memory | ≤16 GB | ~4 GB peak |
| Compute | CPU only | ✅ |
| Network | None | ✅ (fully offline) |

## Project Structure

```
redrob-ranker/
├── rank.py                   # Main entry point
├── src/
│   ├── config.py             # Constants, weights, keyword lists
│   ├── jd_parser.py          # JD understanding & TF-IDF
│   ├── filters.py            # Stage 1: Hard filters
│   ├── features.py           # Stage 2: Feature engineering
│   ├── scorer.py             # Stage 3: Score fusion
│   ├── honeypot.py           # Stage 4: Honeypot detection
│   └── reasoning.py          # Stage 5: Reasoning generation
├── requirements.txt          # Dependencies
├── Dockerfile                # Container for sandbox
├── submission_metadata.yaml  # Hackathon metadata
└── README.md                 # This file
```

## Docker

```bash
docker build -t redrob-ranker .
docker run -v $(pwd)/data:/data redrob-ranker \
    python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
```

## Technologies

- **Python 3.11** — Core language
- **scikit-learn** — TF-IDF vectorization for semantic matching
- **pandas / numpy** — Data manipulation
- **No external LLM APIs** — Fully offline, no GPU needed

## Author

**Sathvik V** — Solo participant, Redrob Hackathon 2026
