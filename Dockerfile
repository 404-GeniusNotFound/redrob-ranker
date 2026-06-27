FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY rank.py .
COPY src/ ./src/

# Default: show help
CMD ["python", "rank.py", "--help"]

# Usage:
# docker build -t redrob-ranker .
# docker run -v $(pwd)/data:/data redrob-ranker \
#     python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
