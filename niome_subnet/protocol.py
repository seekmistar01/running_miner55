"""
Protocol definitions for the Drug Response Prediction Subnet.

This module defines the communication protocols between validators and miners
for drug response prediction tasks using synthetic genomic data.
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import bittensor as bt


@dataclass
class DrugResponsePrediction:
    """Drug response prediction with confidence and signature."""
    predicted_response: str  # "High Efficacy", "Low Efficacy", "Adverse Reaction"
    confidence: float  # Confidence score 0.0-1.0
    signature: str  # Cryptographic signature
    model_version: str = "1.0"


@dataclass
class GroundTruthLabel:
    """Ground truth label from PharmaCAT analysis."""
    match: str  # Allele information
    phenotype: str  # Clinical phenotype
    canonical_phenotype: str  # Canonical phenotype
    drug_name: str  # Drug name


class GenomicsTaskSynapse(bt.Synapse):
    """Protocol for genomics simulation tasks."""
    
    # Input fields
    task: Dict[str, Any]
    timeout: Optional[float] = None  # Timeout window for submission
    
    # Output fields
    vcf_content: Optional[str] = None
    answer_json: Optional[Dict[str, Any]] = None  # Structured answer JSON
    signature: Optional[str] = None  # Cryptographic signature
    probabilities: Optional[List[float]] = None  # Probability vector for χ² test
    error: Optional[str] = None

    def deserialize(self) -> Tuple[Optional[str], Optional[str]]:
        """Deserialize VCF content and error."""
        return self.vcf_content, self.error

    def get_task_json(self) -> str:
        """Get task as JSON string."""
        return json.dumps(self.task, indent=2)
    
    def get_answer_json(self) -> Optional[str]:
        """Get answer as JSON string."""
        if self.answer_json:
            return json.dumps(self.answer_json, indent=2)
        return None