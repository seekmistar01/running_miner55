from typing import List, Tuple, Dict

# ---- General Constants -----

TESTNET_UID = 289
MAINNET_UID = 0                                 # Bittensor subnet uid => shoulde be changed

FORWARD_TIMEOUT = 60                      # Might be optimized further
MINER_QUERY_K = 5                               # Might be optimized further

GENOMIC_TASK_URL = ""                           # <=== Task gerneration, may be https://genomic.io
GENOMIC_STORAGE_URL = ""                        # <=== Data storage url, may be https://genomic.io


# ---- Scoring Constants -----

SCORE_EMA_ALPHA = 0.8
TOP_MINER_COUNT = 10
TOP_MIN_ALPHA_SCALE = 50
TOP_MIN_ALPHA = 0.7
SCORE_DISTRIBUTION = {
    1: 0.60,   # 60%
    2: 0.15,   # 15%
    3: 0.065,   # 6.5%
    4: 0.05,   # 5%
    5: 0.04,   # 4%
    6: 0.025,  # 2.5%
    7: 0.02,   # 2%
    8: 0.02,  # 2%
    9: 0.015,   # 1.5%
    10: 0.015   # 1.5%
}

# ---- Genomic Constants -----

MIN_VCF_SIZE = 100 # Minimum VCF content size in characters

METADATA_VALIDATION_THRESHOLD = 0.8 # Minimum metadata validation score to be considered valid (80%)
PHARMCAT_SCORE_WEIGHT = 0.8 # Weight for PharmaCAT validation score (80%)
METADATA_SCORE_WEIGHT = 0.2 # Weight for metadata validation score (20%)

NON_CAUSAL_SNP_CHECK_LIMIT = 10  # Number of variants to check for non-causal SNPs
MAX_NON_CAUSAL_SNPS = 5 # Maximum number of non-causal SNPs to return

VCF_FILEFORMAT_PREFIX = "##fileformat="
VCF_CHROM_PREFIX = "#CHROM"
VCF_METADATA_KEYS = {
    "population_model": "population_model",
    "population": "population",
    "genome_model": "genome_model",
    "chromosome": "chromosome",
    "miner_hotkey": "miner_hotkey",
}

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
PHARMACOGENE_REGIONS = {
    "chr10": range(96610921, 96724143),  # CYP2C9
    "chr22": range(42522500, 42523270),  # CYP2D6
}

DRUGS = ["warfarin", "clopidogrel", "simvastatin", "metoprolol", "tamoxifen"]

# ---- Backend Request -----
BASE_URL = "http://localhost:5000"