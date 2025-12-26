from pydantic import BaseModel
from typing import Generic, TypeVar

class GenomicSimulationTask(BaseModel):
    id : str
    simulator: str
    population_model: str
    population: str
    genome_model: str
    chromosome: int
    output: str = "vcf"

class GroundTruthLabel(BaseModel):
    """Ground truth label from PharmaCAT analysis."""

    match: str  # Allele information
    phenotype: str  # Clinical phenotype
    canonical_phenotype: str  # Canonical phenotype
    drug_name: str  # Drug name

class ValidationContext:
    """Container for all validation metadata."""
    miner_uid: int
    miner_hotkey: str
    validator_uid: int
    validator_hotkey: str

    def __init__(self, miner_uid: int = 0, miner_hotkey: str = "", validator_uid: int = 0, validator_hotkey: str = ""):
        self.miner_uid = miner_uid
        self.miner_hotkey = miner_hotkey
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey

class TaskPayload(BaseModel):
    """Payload structure for task generation requests."""
    timestamp: float
    hotkey: str
    uuid: str
    netuid: str


PayloadType = TypeVar('PayloadType', bound=BaseModel)

class SignedRequest(BaseModel, Generic[PayloadType]):
    """Generic signed request structure."""
    payload: PayloadType
    payload_raw: str
    signature: str

class MinerScoreDto:
    """Data transfer object for miner score submission."""
    task_id: str
    miner_hotkey: str
    score: str
    weight: str
    file_path: str  # s3 path

    def __init__(self, task_id: str = "", miner_hotkey: str = "", score: str = "", weight: str = "", file_path: str = ""):
        self.task_id = task_id
        self.miner_hotkey = miner_hotkey
        self.score = score
        self.weight = weight
        self.file_path = file_path

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "miner_hotkey": self.miner_hotkey,
            "score": self.score,
            "weight": self.weight,
            "file_path": self.file_path,
        }