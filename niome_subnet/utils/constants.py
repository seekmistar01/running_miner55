# ---- General Constants -----

TESTNET_UID = 289
MAINNET_UID = 55                                # Bittensor subnet uid => shoulde be changed

FORWARD_TIMEOUT = 180                           # Might be optimized further
RESPONSE_TIMEOUT = 60                           # Might be optimized further
MINER_QUERY_K = 5                               # Might be optimized further

GENOMIC_TASK_URL = ""                           # <=== Task gerneration, may be https://genomic.io
GENOMIC_STORAGE_URL = ""                        # <=== Data storage url, may be https://genomic.io


# ---- Scoring Constants -----

SCORE_EMA_ALPHA = 0.8
TOP_MINER_COUNT = 10
TOP_MIN_ALPHA_SCALE = 50
TOP_MIN_ALPHA = 0.7
SCORE_DISTRIBUTION = [0.6, 0.15, 0.065, 0.05, 0.04, 0.025, 0.02, 0.02, 0.015, 0.015]

# ---- Genomic Constants -----
DOCKER_IMAGE = "pgkb/pharmcat"
DOCKER_TIMEOUT = 300  # seconds

MIN_VCF_SIZE = 100 # Minimum VCF content size in characters

METADATA_VALIDATION_THRESHOLD = 0.8 # Minimum metadata validation score to be considered valid (80%)
PHARMCAT_SCORE_WEIGHT = 0.6 # Weight for PharmaCAT validation score (60%)
METADATA_SCORE_WEIGHT = 0.1 # Weight for metadata validation score (10%)
ELAPSED_TIME_WEIGHT = 0.3 # Weight for elapsed time in final score (30%)

VCF_FILEFORMAT_PREFIX = "##fileformat="
VCF_CHROM_PREFIX = "#CHROM"
VCF_METADATA_KEYS = ["population_model", "population", "genome_model", "gene_drugs"]
VCF_PREPROCESSED_MIN_LINES = 36  # Minimum number of lines expected in preprocessed VCF to consider it valid

PHENOTYPES = [
    "Poor Metabolizer",
    "Intermediate Metabolizer",
    "Normal Metabolizer",
    "Rapid Metabolizer",
]
CANONICAL_PHENOTYPES = ["Poor", "Intermediate", "Normal", "Rapid"]
ALLELES = [
    "*1/*1",
    "*1/*2",
    "*2/*2",
    "*1/*3",
    "*2/*3",
    "*3/*3",
]

# ---- Backend Request -----
BASE_URL = "https://niome-api.genomes.io"
MINER_SCORE_URL = f"{BASE_URL}/api/miner_scores"
GET_TASK_URL = f"{BASE_URL}/api/tasks"
S3_UPLOAD_URL = f"{BASE_URL}/api/s3"

# ---- Timeout Values -----
TASK_REQUEST_TIMEOUT = 10  # seconds
FORWARD_REQUEST_TIMEOUT = 30  # seconds
BASE_DELAY_SECONDS = 2  # seconds
SUBMIT_REQUEST_TIMEOUT = 30  # seconds
PHARMCAT_TIMEOUT = 60  # seconds

# ---- Other Constants -----
MAX_TASK_RETRIES = 3
MAX_SUBMIT_RETRIES = 3

CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_CONCURRENT_UPLOADS = 5
MAX_CHUNK_UPLOAD_RETRIES = 3

WITH_DOCKER = True

WANDB_MAX_LOGS = 60_000

SCORING_SYSTEM = "linear"  # "linear", "top"
BURNING_RATE = 1.0
OWNER_HOTKEY = "5DJ5fT174AY8GzbYHnamYQCJd4cTcj2Zf7ogUvBhry1KfYVd"

WEIGHTS_S3_URL = "https://niome-vcf-bucket.s3.us-east-1.amazonaws.com/weights.json"

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = ""
