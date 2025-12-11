from pydantic import BaseModel

class GenomicSimulationTask(BaseModel):
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
