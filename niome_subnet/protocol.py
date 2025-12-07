"""
Protocol definitions for the Drug Response Prediction Subnet.

This module defines the communication protocols between validators and miners
for drug response prediction tasks using synthetic genomic data.
"""

import json
import bittensor as bt
from typing import Dict, Any, Optional

from niome_subnet.genomics.model import GenomicSimulationTask

class GroundTruthLabel:
    """Ground truth label from PharmaCAT analysis."""
    match: str  # Allele information
    phenotype: str  # Clinical phenotype
    canonical_phenotype: str  # Canonical phenotype
    drug_name: str  # Drug name


class GenomicsTaskSynapse(bt.Synapse):
    """Protocol for genomics simulation tasks."""

    # Input fields
    task: Optional[GenomicSimulationTask] = None
    timeout: Optional[float] = None  # Timeout window for submission

    # Output fields
    vcf_content: Optional[str] = None
    answer_json: Optional[Dict[str, Any]] = None  # Structured answer JSON
    signature: Optional[str] = None  # Cryptographic signature

    def deserialize(self) -> bt.Synapse:
        """Deserialize the GenomicsTaskSynapse Object."""
        return self