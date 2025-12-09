import random
from typing import Dict, Any, List

import bittensor as bt
from niome_subnet.genomics.model import GenomicSimulationTask
from niome_subnet.protocol import GenomicsTaskSynapse, GroundTruthLabel
from niome_subnet.genomics.pharmcat_validator import PharmCATValidator
from niome_subnet.utils.constants import (
    MIN_VCF_SIZE, 
    METADATA_VALIDATION_THRESHOLD, 
    PHARMCAT_SCORE_WEIGHT, 
    METADATA_SCORE_WEIGHT, 
    MAX_NON_CAUSAL_SNPS, 
    NON_CAUSAL_SNP_CHECK_LIMIT,
    VCF_CHROM_PREFIX,
    VCF_FILEFORMAT_PREFIX,
    VCF_METADATA_KEYS,
    PHENOTYPES,
    CANONICAL_PHENOTYPES,
    PHARMACOGENE_REGIONS,
    DRUGS,
    ALLELES
)   

def validate_response(response: GenomicsTaskSynapse, task: GenomicSimulationTask) -> float:
    drug_name = random.choice(DRUGS)
    miner_vcf = response.vcf_content
    
    if not _is_vcf_valid(miner_vcf):
        return 0.0
    
    metadata_score = _compute_metadata_score(miner_vcf, task)
    pharmcat_score = _compute_pharmcat_score(miner_vcf, drug_name)
    final_score = (PHARMCAT_SCORE_WEIGHT * pharmcat_score) + (METADATA_SCORE_WEIGHT * metadata_score)
    bt.logging.info(f"Validation scores - Metadata: {metadata_score:.4f}, PharmCAT: {pharmcat_score:.4f}, Final: {final_score:.4f}")
    print(f"Validation scores - Metadata: {metadata_score:.4f}, PharmCAT: {pharmcat_score:.4f}, Final: {final_score:.4f}")
    return final_score
    
def _is_vcf_valid(miner_vcf: str) -> bool:
    validation = _validate_miner_vcf(miner_vcf)
    if not validation.get("valid", False):
        bt.logging.warning(f"VCF validation failed: {validation.get('errors', [])}")
        return False
    return True

def _compute_metadata_score(miner_vcf: str, task: GenomicSimulationTask) -> float:
    validation = _validate_vcf_metadata(miner_vcf, task)
    return validation.get("score", 0.0)

def _compute_pharmcat_score(miner_vcf: str, drug_name: str) -> float:
    validator = PharmCATValidator()
    pharmcat_results = _run_pharmcat(miner_vcf, drug_name, validator)
    non_causal_snps = _identify_non_causal_snps(miner_vcf)
    validation_result = validator.validate_miner_response(
        miner_vcf, drug_name, non_causal_snps, pharmcat_results
    )
    return validation_result.get("validation_score", 0.0)

# VCF Validation Functions
def _validate_miner_vcf(miner_vcf: str) -> Dict[str, Any]:
    validation_result = {
        "valid": False,
        "errors": [],
        "warnings": []
    }
    try:
        validation_result = _has_minimum_size(miner_vcf, validation_result)
        lines = miner_vcf.splitlines()
        
        has_fileformat = _validate_fileformat_header(lines, validation_result)        
        has_chrom_header = _validate_chrom_header(lines, validation_result)

        # Check for data lines
        data_lines = [line for line in lines if line.strip() and not line.startswith("#")]
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
        
    except Exception as e:
        validation_result["errors"].append(f"VCF validation error: {str(e)}")
    return validation_result

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

    result["errors"].append("Missing #CHROM header line.")
    return False

def _validate_data_lines(data_lines: List[str], result: Dict[str, Any]) -> None:
    if not data_lines:
        result["warnings"].append("No data lines found in VCF.")
    elif len(data_lines) < 1:
        result["warnings"].append("Very few data lines in VCF.")

def _validate_header_count(header_lines: List[str], result: Dict[str, Any]) -> None:
    if len(header_lines) < 3:
        result["warnings"].append("Very few header lines in VCF.")

# VCF Metadata Validation Functions
def _validate_vcf_metadata(miner_vcf: str, task: GenomicSimulationTask) -> Dict[str, Any]:
    validation_result = {
        "valid": False,
        "extracted_metadata": [],
        "mismatches": [],
        "score" : 0.0,
        "details" : {}
    }

    try:
        extracted_metadata = _extract_vcf_metadata(miner_vcf)
        validation_result["extracted_metadata"] = extracted_metadata
        if not extracted_metadata:
            validation_result["mismatches"].append("No metadata extracted from VCF headers")
            return validation_result
        validation_result = _compare_task_metadata(extracted_metadata, validation_result, task)
    except Exception as e:
        validation_result["mismatches"].append(f"Error extracting metadata: {str(e)}")

    return validation_result

def _extract_vcf_metadata(miner_vcf: str) -> Dict[str, Any]:
    metadata = {}    
    try:
        if not miner_vcf or not miner_vcf.strip():
            return metadata
        
        # Find header section (everything before #CHROM line)
        header_section = _extract_header_section(miner_vcf)
        
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
                    
                    # Map to standard property names
                    if key == "population_model":
                        metadata["population_model"] = value
                    elif key == "population":
                        metadata["population"] = value
                    elif key == "genome_model":
                        metadata["genome_model"] = value
                    elif key == "chromosome":
                        metadata["chromosome"] = value
                    elif key == "allele_one":
                        metadata["allele_one"] = value
                    elif key == "allele_two":
                        metadata["allele_two"] = value
    
    except Exception as e:
        bt.logging.warning(f"Error extracting VCF metadata: {e}")
    
    return metadata

def _extract_header_section(vcf: str) -> str:
    chrom_idx = vcf.find(VCF_CHROM_PREFIX)
    if chrom_idx == -1:
        return vcf  # treat entire file as header

    return vcf[:chrom_idx]

def _compare_task_metadata(extracted_metadata: Dict[str, Any], validation_result:Dict[str, Any], task: GenomicSimulationTask) -> Dict[str, Any]:
    checks_passed = 0
    checks_total = 0
    
    for key, metadata_key in VCF_METADATA_KEYS.items():
        if key not in task:
            continue

        checks_total += 1
        expected = str(task[key]).strip()
        actual = str(extracted_metadata.get(metadata_key, "")).strip()

        if _metadata_match(key, expected, actual):
            checks_passed += 1
            validation_result["details"][metadata_key] = "match"
        else:
            validation_result["details"][metadata_key] = "mismatch"
            validation_result["mismatches"].append(
                f"{metadata_key}: expected '{expected}', got '{actual}'"
            )
   
    # score
    if checks_total > 0:
        validation_result["score"] = checks_passed / checks_total
        validation_result["valid"] = validation_result["score"] >= METADATA_VALIDATION_THRESHOLD

    # bonus field check
    validation_result["details"]["has_miner_hotkey"] = "miner_hotkey" in extracted_metadata
    if "miner_hotkey" not in extracted_metadata:
        validation_result["mismatches"].append("Missing miner_hotkey in metadata")
    
    return validation_result

def _metadata_match(key: str, expected: str, actual: str) -> bool:
    """Special comparison rules depending on metadata key."""

    # Normalize chromosome (e.g., "chr1", "CHR1", "1")
    if key == "chromosome":
        e = expected.lower().replace("chr", "")
        a = actual.lower().replace("chr", "")
        return e == a

    # Case-insensitive compare
    return expected.lower() == actual.lower()

# PharmCAT Validation Functions
def _run_pharmcat(miner_vcf: str, drug_name: str, pharmcat_validator: PharmCATValidator) -> Dict[str, Any]:
    try:
        pharmcat_results = pharmcat_validator.get_ground_truth(miner_vcf, drug_name)
        return pharmcat_results
    except Exception as e:
        bt.logging.error(f"PharmCAT error: {e}")
        synthetic_gt = _generate_synthetic_ground_truth(drug_name)
        return {
                "match": {"alleles": synthetic_gt.match},
                "phenotype": {
                    "clinical_call": synthetic_gt.phenotype,
                    "canonical_phenotype": synthetic_gt.canonical_phenotype
                }
        }

def _generate_synthetic_ground_truth(drug_name: str) -> GroundTruthLabel:    
    return GroundTruthLabel(
        match=random.choice(ALLELES),
        phenotype=random.choice(PHENOTYPES),
        canonical_phenotype=random.choice(CANONICAL_PHENOTYPES),
        drug_name=drug_name
    )

def extract_gold_label(pharmcat_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract canonical phenotype/dose as gold label from PharmCAT results.
    
    Args:
        pharmcat_results: PharmCAT analysis results
        
    Returns:
        Dict containing canonical phenotype and dose recommendations
    """
    if "error" in pharmcat_results:
        return {"error": pharmcat_results["error"], "phenotype": None, "dose": None}
    
    try:
        phenotype_data = pharmcat_results.get("phenotype", {})
        match_data = pharmcat_results.get("match", {})
        
        gold_label = {
            "phenotype": phenotype_data.get("canonical_phenotype", "Unknown"),
            "dose": phenotype_data.get("dose_recommendation", {}),
            "key_alleles": match_data.get("key_alleles", []),
            "confidence": phenotype_data.get("confidence", 0.0)
        }
        
        bt.logging.info(f"Extracted gold label: {gold_label}")
        return gold_label
        
    except Exception as e:
        bt.logging.error(f"Error extracting gold label: {str(e)}")
        return {"error": str(e), "phenotype": None, "dose": None}
    
def _identify_non_causal_snps(vcf_content: str) -> List[str]:
    """
    Identify non-causal SNPs for adversarial testing.

    This is a simplified implementation - in practice would use
    pharmacogenomic knowledge bases to identify non-causal variants.
    """
    non_causal_snps = []
    try:
        lines = vcf_content.split('\n')
        data_lines = [line for line in lines if not line.startswith('#') and line.strip()]
        
        for line in data_lines[:NON_CAUSAL_SNP_CHECK_LIMIT]:  # Check limited number of variants
            fields = line.split('\t')
            if len(fields) >= 2:
                chrom = fields[0]
                pos = fields[1]
                
                # Check if position is outside pharmacogene regions
                is_non_causal = True
                if chrom in PHARMACOGENE_REGIONS:
                    try:
                        pos_int = int(pos)
                        if pos_int in PHARMACOGENE_REGIONS[chrom]:
                            is_non_causal = False
                    except ValueError:
                        pass
                
                if is_non_causal:
                    non_causal_snps.append(pos)
    except Exception as e:
        bt.logging.warning(f"Error identifying non-causal SNPs: {e}")

    return non_causal_snps[:MAX_NON_CAUSAL_SNPS]  # Return up to max non-causal SNPs
