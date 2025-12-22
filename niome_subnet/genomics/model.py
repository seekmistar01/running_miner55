from pydantic import BaseModel
from typing import Generic, TypeVar

class GenomicSimulationTask(BaseModel):
    task_id : str
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
    signature: str


TaskRequest = SignedRequest[TaskPayload]
    