from typing import List

import bittensor as bt

from niome_subnet.genomics.pharmcat_validator import PharmCATValidator
from niome_subnet.utils.constants import (
    MAX_NON_CAUSAL_SNPS,  
    NON_CAUSAL_SNP_CHECK_LIMIT,
    PHARMACOGENE_REGIONS,
)

def compute_pharmcat_score(miner_vcf: str) -> float:
    try:
        validator = PharmCATValidator()
        pharmcat_results = validator.get_ground_truth(miner_vcf)
        non_causal_snps = _identify_non_causal_snps(miner_vcf)
        validation_result = validator.validate_miner_response(
            miner_vcf, non_causal_snps, pharmcat_results
        )
        return validation_result.get("validation_score", 0.0)
    except Exception as e:
        bt.logging.error(f"Error computing PharmCAT score: {e}")
        return 0.0

def _identify_non_causal_snps(vcf_content: str) -> List[str]:
    """
    Identify non-causal SNPs for adversarial testing.

    This is a simplified implementation - in practice would use
    pharmacogenomic knowledge bases to identify non-causal variants.
    """
    non_causal_snps = []
    try:
        lines = vcf_content.split("\n")
        data_lines = [
            line for line in lines if not line.startswith("#") and line.strip()
        ]

        for line in data_lines[
            :NON_CAUSAL_SNP_CHECK_LIMIT
        ]:  # Check limited number of variants
            fields = line.split("\t")
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
