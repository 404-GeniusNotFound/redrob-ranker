"""
config.py — Central configuration for the Redrob Ranker.

All weights, thresholds, keyword lists, and scoring constants live here.
Tuned based on careful reading of the JD for Senior AI Engineer — Founding Team.
"""

from datetime import date

# ============================================================================
# Reference date for recency calculations
# ============================================================================
REFERENCE_DATE = date(2026, 6, 15)  # Approximate "now" for signal calculations

# ============================================================================
# Scoring dimension weights (must sum to 1.0)
# ============================================================================
WEIGHTS = {
    "career_coherence": 0.25,     # Career trajectory & company-type fit
    "skill_authenticity": 0.25,   # Genuine skill depth & coverage
    "jd_semantic_align": 0.20,    # Text-level alignment with JD concepts
    "behavioral_avail": 0.15,     # Platform engagement & availability
    "location_fit": 0.10,         # Geographic match
    "education_cred": 0.05,       # Education + credibility signals
}

# ============================================================================
# Stage 1 — Hard filter thresholds
# ============================================================================
MIN_EXPERIENCE_YEARS = 2.0
MAX_EXPERIENCE_YEARS = 20.0

# ============================================================================
# JD-derived "must have" skill concepts
# These are the CONCEPTS the JD cares about — mapped to various surface forms
# ============================================================================
MUST_HAVE_SKILL_CONCEPTS = {
    "embeddings_retrieval": [
        "embeddings", "sentence-transformers", "sentence transformers",
        "openai embeddings", "bge", "e5", "retrieval", "search",
        "information retrieval", "dense retrieval", "hybrid retrieval",
        "semantic search", "rag", "retrieval augmented", "embedding",
    ],
    "vector_databases": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "vector database", "vector db",
        "vector search", "ann", "approximate nearest neighbor",
        "hybrid search", "chromadb", "chroma",
    ],
    "python": [
        "python", "pytorch", "tensorflow", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "fastapi", "flask", "django",
    ],
    "evaluation_frameworks": [
        "ndcg", "mrr", "map", "precision", "recall", "f1",
        "a/b test", "ab test", "evaluation", "benchmark", "metrics",
        "offline evaluation", "online evaluation", "ranking evaluation",
        "mean average precision", "mean reciprocal rank",
    ],
    "ranking_recommendation": [
        "ranking", "recommendation", "recommender", "learning to rank",
        "learning-to-rank", "l2r", "re-ranking", "reranking",
        "candidate matching", "search ranking", "relevance",
    ],
}

NICE_TO_HAVE_SKILL_CONCEPTS = {
    "llm_finetuning": [
        "fine-tuning", "fine tuning", "finetuning", "lora", "qlora",
        "peft", "llm", "large language model", "gpt", "llama",
        "mistral", "instruction tuning", "rlhf", "dpo",
    ],
    "learning_to_rank": [
        "xgboost", "lightgbm", "gradient boosting", "learning to rank",
        "lambdamart", "catboost", "listwise", "pairwise", "pointwise",
    ],
    "hrtech_marketplace": [
        "hr tech", "hrtech", "recruiting", "recruitment", "talent",
        "marketplace", "ats", "applicant tracking",
    ],
    "distributed_systems": [
        "distributed", "kubernetes", "k8s", "docker", "microservices",
        "kafka", "spark", "ray", "dask", "inference optimization",
        "model serving", "triton", "onnx", "tensorrt",
    ],
    "open_source": [
        "open source", "open-source", "github", "contributor",
        "contributions", "oss",
    ],
}

# Negative signals — JD's "what we do NOT want"
NEGATIVE_SKILL_CONCEPTS = {
    "computer_vision_only": [
        "computer vision", "image segmentation", "object detection",
        "yolo", "resnet", "cnn", "convolutional",
    ],
    "speech_robotics": [
        "speech recognition", "asr", "tts", "text to speech",
        "robotics", "ros", "autonomous driving", "lidar",
    ],
}

# ============================================================================
# Company type classification
# ============================================================================
CONSULTING_SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcl technologies", "tech mahindra", "mindtree", "l&t infotech",
    "lti", "ltimindtree", "mphasis", "cyient", "hexaware", "zensar",
    "persistent", "persistent systems", "niit technologies", "coforge",
    "birlasoft", "sonata software", "sasken", "ness digital",
    "deloitte", "ey", "ernst & young", "kpmg", "pwc",
    "pricewaterhousecoopers", "mckinsey", "bain", "bcg",
    "boston consulting", "oliver wyman",
}

# Known product companies (positive signal)
PRODUCT_COMPANIES_KEYWORDS = [
    "google", "microsoft", "amazon", "meta", "facebook", "apple",
    "netflix", "uber", "lyft", "airbnb", "stripe", "shopify",
    "flipkart", "swiggy", "zomato", "razorpay", "cred", "meesho",
    "paytm", "phonepe", "groww", "zerodha", "byju", "unacademy",
    "ola", "dunzo", "freshworks", "zoho", "postman", "browserstack",
    "hasura", "chargebee", "clevertap", "moengage", "linkedin",
    "twitter", "x corp", "salesforce", "adobe", "atlassian", "databricks",
    "snowflake", "datadog", "confluent", "elastic", "mongodb",
    "openai", "anthropic", "cohere", "huggingface", "hugging face",
    "stability ai", "redrob",
]

# ============================================================================
# Title / role classification
# ============================================================================
# Titles that strongly suggest tech/engineering/AI background
TECH_TITLE_KEYWORDS = [
    "engineer", "developer", "programmer", "architect", "scientist",
    "researcher", "analyst", "data", "ml", "ai", "machine learning",
    "artificial intelligence", "deep learning", "nlp", "software",
    "backend", "frontend", "full stack", "fullstack", "devops", "sre",
    "platform", "infrastructure", "cloud", "sde", "swe",
    "technical lead", "tech lead", "cto", "vp engineering",
]

# Titles that strongly suggest NON-tech background
NON_TECH_TITLES = [
    "accountant", "hr manager", "human resources", "marketing manager",
    "sales executive", "sales manager", "graphic designer",
    "content writer", "customer support", "operations manager",
    "civil engineer", "mechanical engineer", "chemical engineer",
    "electrical engineer", "finance", "legal", "compliance",
    "procurement", "supply chain", "logistics",
]

# ============================================================================
# Career coherence scoring parameters
# ============================================================================
IDEAL_AVG_TENURE_MONTHS = 30     # ~2.5 years per role
MIN_ACCEPTABLE_TENURE_MONTHS = 18  # < 1.5 years = title-chaser warning
TITLE_CHASER_PENALTY = 0.3        # Multiplier if avg tenure < 18 months

# ============================================================================
# Behavioral availability signal weights (within the 15pt dimension)
# ============================================================================
BEHAVIORAL_WEIGHTS = {
    "open_to_work": 0.10,
    "recency": 0.25,              # last_active_date
    "response_rate": 0.25,
    "response_time": 0.10,
    "notice_period": 0.10,
    "interview_completion": 0.10,
    "profile_completeness": 0.05,
    "offer_acceptance": 0.05,
}

# ============================================================================
# Location scoring
# ============================================================================
# JD says: Pune/Noida preferred, Hyderabad/Mumbai/Delhi NCR welcome
PREFERRED_LOCATIONS = {"pune", "noida"}
WELCOME_LOCATIONS = {
    "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram",
    "new delhi", "bangalore", "bengaluru", "chennai", "kolkata",
}

# ============================================================================
# Honeypot detection thresholds
# ============================================================================
HONEYPOT_MAX_EXPERT_SKILLS_ZERO_DURATION = 3  # >3 "expert" skills with <=3 months
HONEYPOT_EXPERIENCE_DATE_MISMATCH_MONTHS = 24  # yrs_exp vs career dates off by >2 yrs
HONEYPOT_TITLE_DESC_MISMATCH_THRESHOLD = 0.15  # Very low TF-IDF similarity

# ============================================================================
# JD full text (embedded for TF-IDF — avoids file I/O dependency)
# ============================================================================
JD_CORE_TEXT = """
Senior AI Engineer founding team. Series A AI-native talent intelligence platform.
Own the intelligence layer. Ranking retrieval matching systems.
Ship a v2 ranking system. Embeddings hybrid retrieval LLM-based re-ranking.
Evaluation infrastructure offline benchmarks online A/B testing recruiter feedback loops.
Production experience embeddings-based retrieval systems sentence-transformers.
Embedding drift index refresh retrieval quality regression production.
Vector databases hybrid search infrastructure Pinecone Weaviate Qdrant Milvus
OpenSearch Elasticsearch FAISS.
Strong Python code quality.
Evaluation frameworks ranking systems NDCG MRR MAP offline online correlation A/B test.
LLM fine-tuning LoRA QLoRA PEFT.
Learning-to-rank models XGBoost neural.
HR-tech recruiting marketplace products.
Distributed systems large-scale inference optimization.
Open-source contributions AI ML space.
Scrappy product-engineering attitude ship working ranker.
Candidate-JD matching at scale mentoring.
5 to 9 years experience applied ML AI product companies not pure services.
Shipped end-to-end ranking search recommendation system real users meaningful scale.
Strong opinions retrieval hybrid vs dense evaluation offline vs online LLM integration
fine-tune vs prompt.
"""

# JD negative concepts (things the JD says it does NOT want)
JD_NEGATIVE_TEXT = """
Title chasers switching companies every 1.5 years.
Framework enthusiasts LangChain tutorials demo projects.
Only consulting firms TCS Infosys Wipro Accenture Cognizant Capgemini.
Computer vision speech robotics without NLP IR exposure.
Pure research academic labs no production deployment.
AI experience only recent LangChain calling OpenAI.
Senior engineer not written production code 18 months.
Architecture tech lead roles no coding.
"""

# ============================================================================
# Scoring normalization
# ============================================================================
MAX_POINTS = {
    "career_coherence": 25.0,
    "skill_authenticity": 25.0,
    "jd_semantic_align": 20.0,
    "behavioral_avail": 15.0,
    "location_fit": 10.0,
    "education_cred": 5.0,
}

# ============================================================================
# Education field relevance
# ============================================================================
RELEVANT_EDUCATION_FIELDS = [
    "computer science", "computer engineering", "software engineering",
    "information technology", "data science", "machine learning",
    "artificial intelligence", "statistics", "mathematics",
    "electrical engineering", "electronics", "ece",
    "computational", "informatics",
]
