from typing import List, Dict, Any

import bittensor as bt

from niome_subnet.utils.constants import MIN_VCF_SIZE, VCF_FILEFORMAT_PREFIX, VCF_CHROM_PREFIX

def is_vcf_valid(miner_vcf: str) -> bool:
    # Validate miner vcf
    validation_result = {"valid": False, "errors": [], "warnings": []}
    try:
        validation_result = _has_minimum_size(miner_vcf, validation_result)
        lines = miner_vcf.splitlines()

        has_fileformat = _validate_fileformat_header(lines, validation_result)
        has_chrom_header = _validate_chrom_header(lines, validation_result)

        # Check for data lines
        data_lines = [
            line for line in lines if line.strip() and not line.startswith("#")
        ]
        header_lines = [line for line in lines if line.startswith("##")]

        _validate_data_lines(data_lines, validation_result)
        _validate_header_count(header_lines, validation_result)
        # Determine if valid
        validation_result["valid"] = (
            has_fileformat
            and has_chrom_header
            and len(data_lines) > 0
            and not validation_result["errors"]
        )

        if not validation_result.get("valid", False):
            bt.logging.warning(f"VCF validation failed: {validation_result.get('errors', [])}")
            return False
        return True

    except Exception as e:
        bt.logging.error(f"Error during VCF validation: {e}")
        return False

def _has_minimum_size(vcf: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if not vcf or len(vcf.strip()) < MIN_VCF_SIZE:
        result["errors"].append("VCF content is too short or empty.")
    return result


def _validate_fileformat_header(lines: List[str], result: Dict[str, Any]) -> bool:
    for line in lines:
        if line.startswith(VCF_FILEFORMAT_PREFIX):
            if "VCF" not in line.upper():
                result["warnings"].append("VCF fileformat may be invalid.")
            return True

    result["errors"].append("Missing ##fileformat header.")
    return False


def _validate_chrom_header(lines: List[str], result: Dict[str, Any]) -> bool:
    for line in lines:
        if line.startswith(VCF_CHROM_PREFIX):
            return True

    result["errors"].append(f"Missing {VCF_CHROM_PREFIX} header line.")
    return False


def _validate_data_lines(data_lines: List[str], result: Dict[str, Any]) -> None:
    if not data_lines:
        result["warnings"].append("No data lines found in VCF.")
    elif len(data_lines) < 1:
        result["warnings"].append("Very few data lines in VCF.")


def _validate_header_count(header_lines: List[str], result: Dict[str, Any]) -> None:
    if len(header_lines) < 3:
        result["warnings"].append("Very few header lines in VCF.")
