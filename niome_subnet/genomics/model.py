from pydantic import BaseModel

class GenomicSimulationTask(BaseModel):
    simulator: str
    population_model: str
    population: str
    genome_model: str
    chromosome: int
    output: str = "vcf"