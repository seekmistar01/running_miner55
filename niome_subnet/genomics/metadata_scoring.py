import ast
from typing import Dict, Any

import bittensor as bt

from niome_subnet.genomics.model import GenomicSimulationTask
from niome_subnet.utils.constants import (
    VCF_METADATA_KEYS,
    METADATA_VALIDATION_THRESHOLD,
    VCF_CHROM_PREFIX,
)


def compute_metadata_score(miner_vcf: str, task: GenomicSimulationTask) -> float:
    validation = {
        "valid": False,
        "extracted_metadata": [],
        "mismatches": [],
        "score": 0.0,
        "details": {},
    }

    try:
        extracted_metadata = _extract_vcf_metadata(miner_vcf)
        validation["extracted_metadata"] = extracted_metadata

        if not extracted_metadata:
            validation["mismatches"].append("No metadata extracted from VCF headers")
        else:
            validation = _compare_task_metadata(extracted_metadata, validation, task)

        return validation.get("score", 0.0)
    except Exception as e:
        validation["mismatches"].append(f"Error extracting metadata: {str(e)}")
        return 0.0


def _extract_vcf_metadata(miner_vcf: str) -> Dict[str, Any]:
    metadata = {}
    try:
        if not miner_vcf or not miner_vcf.strip():
            return metadata

        # Find header section (everything before #CHROM line)
        chrom_idx = miner_vcf.find(VCF_CHROM_PREFIX)
        if chrom_idx == -1:
            header_section = miner_vcf
        header_section = miner_vcf[:chrom_idx]

        # Parse header lines
        header_lines = header_section.splitlines()

        for line in header_lines:
            line = line.strip()
            if not line.startswith("##"):
                continue

            # Parse metadata lines in format: ##key=value
            if "=" in line:
                # Remove ## prefix
                line_content = line[2:] if line.startswith("##") else line
                parts = line_content.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key in VCF_METADATA_KEYS:
                        metadata[key] = value

    except Exception as e:
        bt.logging.warning(f"Error extracting VCF metadata: {e}")

    return metadata


def _compare_task_metadata(
    extracted_metadata: Dict[str, Any],
    validation_result: Dict[str, Any],
    task: GenomicSimulationTask,
) -> Dict[str, Any]:
    checks_passed = 0
    checks_total = 0

    for key in VCF_METADATA_KEYS:
        if key not in task:
            continue

        checks_total += 1
        expected = str(task[key]).strip()
        actual = str(extracted_metadata.get(key, "")).strip()

        if expected.lower() == actual.lower():
            checks_passed += 1
            validation_result["details"][key] = "match"
        else:
            validation_result["details"][key] = "mismatch"
            validation_result["mismatches"].append(
                f"{key}: expected '{expected}', got '{actual}'"
            )

    # score
    if checks_total > 0:
        validation_result["score"] = checks_passed / checks_total
        validation_result["valid"] = (
            validation_result["score"] >= METADATA_VALIDATION_THRESHOLD
        )

    # bonus field check
    validation_result["details"]["has_miner_hotkey"] = (
        "miner_hotkey" in extracted_metadata
    )
    if "miner_hotkey" not in extracted_metadata:
        validation_result["mismatches"].append("Missing miner_hotkey in metadata")

    return validation_result
