# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 genomes.io
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import tempfile
import os
import subprocess
import glob
from typing import Dict, List, Optional, Any

import bittensor as bt
import pharmcat_runner

from niome_subnet.utils.constants import PHARMCAT_TIMEOUT

class PharmCATValidator:
    """
    PharmCAT-based validation system for pharmacogenomic VCF analysis.
    
    This class implements:
    1. Ground-truth retrieval using PharmCAT
    2. Adversarial QC with SNP swapping
    """
    
    def __init__(self):
        """
        Initialize PharmCAT validator using pharmcat-runner Python package.
        
        Uses Python directly to run PharmaCAT as specified in requirements.
        """
        # Initialize pharmcat-runner Python package
        try:
            # pharmcat_runner is a function-based package, no class initialization needed
            # Verify pharmcat_runner is available
            bt.logging.info("Initialized PharmCAT validator with pharmcat-runner Python package")
        except ImportError:
            bt.logging.warning(
                "pharmcat-runner package not found. "
                "Install with: pip install pharmcat-runner"
            )
            bt.logging.warning("PharmCAT validation will use fallback mode")
        except Exception as e:
            bt.logging.error(f"Failed to initialize pharmcat-runner: {e}")
            bt.logging.warning("PharmCAT validation will use fallback mode")
    
    def get_ground_truth(self, vcf_content: str) -> Dict[str, Any]:
        """
        Get ground truth from PharmCAT for a VCF file and combination.
        
        Args:
            vcf_content: VCF file content as string
            
        Returns:
            Dict containing PharmCAT results with match (alleles) and phenotype (clinical call)
        """
        try:
            # Create temporary VCF file
            with tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False) as temp_vcf:
                temp_vcf.write(vcf_content)
                temp_vcf_path_unsorted = temp_vcf.name

            bt.logging.debug(f"temp saved VCF file: {temp_vcf_path_unsorted}")

            temp_vcf_path = f"{temp_vcf_path_unsorted}.sorted.vcf"

            # Before you can run PharmCAT you need to sort the file using bcftools:
            cmd = ["bcftools", "sort", temp_vcf_path_unsorted]

            with open(temp_vcf_path, "w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, timeout=PHARMCAT_TIMEOUT)

            if result.returncode != 0:
                raise Exception(f"vcf sort failed: {result.stderr}")

            if os.path.exists(temp_vcf_path):
                bt.logging.debug(f"The file {temp_vcf_path} exists.")
            else:
                bt.logging.error(f"The file {temp_vcf_path} does not exist.")

            # Clean up temporary file
            if os.path.exists(temp_vcf_path_unsorted):
                bt.logging.info(f"delete VCF file =: {temp_vcf_path_unsorted}")
                os.unlink(temp_vcf_path_unsorted)

            # Run PharmCAT
            pharmcat_results = self._run_pharmcat(temp_vcf_path)

            # Clean up temporary file
            if os.path.exists(temp_vcf_path):
                bt.logging.info(f"delete VCF file =: {temp_vcf_path}")
                os.unlink(temp_vcf_path)

            # Delete all temp files with the same prefix
            pattern = temp_vcf_path + ".*"
            for f in glob.glob(pattern):
                if os.path.isfile(f):
                    bt.logging.info(f"delete temp VCF files: {f}")
                    os.unlink(f)
            return pharmcat_results
        except Exception as e:
            bt.logging.error(f"Error running PharmCAT: {str(e)}")
            return {"error": str(e), "match": {}, "phenotype": {}}
    
    def _run_pharmcat(self, vcf_path: str) -> Dict[str, Any]:
        """
        Run PharmCAT on VCF file using Python pharmcat-runner.
        
        Args:
            vcf_path: Path to VCF file
        Returns:
            Dict with PharmCAT results containing:
            - match: JSON blob with alleles
            - phenotype: JSON blob with clinical call
        """
        try:
            return self._run_pharmcat_local(vcf_path)
                
        except Exception as e:
            bt.logging.error(f"Error running PharmCAT: {str(e)}")
            return {"error": str(e), "match": {}, "phenotype": {}}
    
    def _run_pharmcat_local(self, vcf_path: str) -> Dict[str, Any]:
        """
        Run PharmCAT using Python pharmcat-runner package.
        
        Receives two JSON blobs:
        - match: alleles information
        - phenotype: clinical call
        
        Args:
            vcf_path: Path to VCF file
            
        Returns:
            Dict with match (alleles) and phenotype (clinical call) JSON blobs
        """
        try:
            # Use pharmcat-runner Python package to execute PharmCAT
            import tempfile
            
            with tempfile.TemporaryDirectory() as tempdir:
                # Run PharmaCAT using Python package
                try:
                    results = pharmcat_runner.run_pharmcat(vcf_path, tempdir)
                except NameError:
                    # pharmcat-runner not imported, try importing it
                    results = pharmcat_runner.run_pharmcat(vcf_path, tempdir)
                
                # Parse and format results according to requirements
                # Requirement: Receive two JSON blobs: match (alleles) and phenotype (clinical call)
                if isinstance(results, dict):
                    # Extract match (alleles) JSON blob
                    match_data = results.get("match", {})
                    if not match_data:
                        # Try alternative keys
                        match_data = results.get("alleles", {})
                    phenotype_data = results.get("phenotypes", {})
                    if not phenotype_data:
                        # Try alternative keys
                        phenotype_data = results.get("clinical_call", {})
                    
                    return {
                        "match": match_data,  # JSON blob with alleles
                        "phenotype": phenotype_data,  # JSON blob with clinical call
                        "raw_output": results
                    }
                else:
                    # Handle case where results might be a different format
                    # Try to extract from file outputs in tempdir
                    return self._parse_pharmcat_files(tempdir)
                
        except ImportError:
            bt.logging.error("pharmcat-runner package not installed")
            return {"error": "pharmcat-runner not installed", "match": {}, "phenotype": {}}
        except Exception as e:
            bt.logging.error(f"Error running PharmCAT with Python: {str(e)}")
            return {"error": str(e), "match": {}, "phenotype": {}}
    
    def _parse_pharmcat_files(self, output_dir: str) -> Dict[str, Any]:
        """
        Parse PharmaCAT output files if direct Python API doesn't return expected format.
        
        Args:
            output_dir: Directory containing PharmaCAT output files
            
        Returns:
            Dict with match and phenotype JSON blobs
        """
        import os
        import json
        
        match_data = {}
        phenotype_data = {}
        
        try:
            # Look for common PharmaCAT output files
            for filename in os.listdir(output_dir):
                filepath = os.path.join(output_dir, filename)
                
                if filename.endswith('.json'):
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            
                            # Try to extract match and phenotype from JSON file
                            if "match" in data or "alleles" in data:
                                match_data.update(data.get("match", data.get("alleles", {})))
                            if "phenotype" in data or "clinical_call" in data:
                                phenotype_data.update(
                                    data.get("phenotype", data.get("clinical_call", {}))
                                )
                    except (json.JSONDecodeError, IOError):
                        continue
                
                elif filename.endswith('.txt') or filename.endswith('.report'):
                    # Try to parse text-based reports
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            # Basic parsing - would need to adapt based on actual PharmaCAT output format
                            if "phenotype" in content.lower():
                                phenotype_data["text_report"] = content
                    except IOError:
                        continue
            
        except Exception as e:
            bt.logging.warning(f"Error parsing PharmaCAT files: {e}")
        
        return {
            "match": match_data,
            "phenotype": phenotype_data,
        }
    
    def extract_gold_label(self, pharmcat_results: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def create_adversarial_vcf(self, original_vcf: str, non_causal_snps: List[str]) -> str:
        """
        Create adversarial VCF by swapping non-causal SNPs.
        
        Args:
            original_vcf: Original VCF content
            non_causal_snps: List of non-causal SNP positions to swap
            
        Returns:
            Modified VCF content with swapped SNPs
        """
        try:
            lines = original_vcf.split('\n')
            modified_lines = []
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    # Keep headers and empty lines unchanged
                    modified_lines.append(line)
                else:
                    # Parse VCF data line
                    fields = line.split('\t')
                    if len(fields) >= 8:  # Basic VCF format check
                        chrom, pos, id_field, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]
                        
                        # Check if this position should be swapped
                        if pos in non_causal_snps:
                            # Swap reference and alternate alleles
                            fields[3], fields[4] = fields[4], fields[3]
                            bt.logging.info(f"Swapped alleles at position {pos}: {ref} <-> {alt}")
                        
                        modified_lines.append('\t'.join(fields))
                    else:
                        modified_lines.append(line)
            
            modified_vcf = '\n'.join(modified_lines)
            bt.logging.info(f"Created adversarial VCF with {len(non_causal_snps)} swapped SNPs")
            return modified_vcf
            
        except Exception as e:
            bt.logging.error(f"Error creating adversarial VCF: {str(e)}")
            return original_vcf
    
    def measure_drift(self, original_results: Dict[str, Any], adversarial_results: Dict[str, Any]) -> float:
        """
        Measure drift between original and adversarial results.
        
        Args:
            original_results: Results from original VCF
            adversarial_results: Results from adversarial VCF
            
        Returns:
            Drift score (0.0 = no drift, 1.0 = maximum drift)
        """
        try:
            if "error" in original_results or "error" in adversarial_results:
                return 1.0  # Maximum penalty for errors
            
            # Compare phenotypes
            orig_phenotype = original_results.get("phenotype", {}).get("canonical_phenotype", "")
            adv_phenotype = adversarial_results.get("phenotype", {}).get("canonical_phenotype", "")
            
            # Compare key alleles
            orig_alleles = set(original_results.get("match", {}).get("key_alleles", []))
            adv_alleles = set(adversarial_results.get("match", {}).get("key_alleles", []))
            
            # Calculate phenotype drift
            phenotype_drift = 0.0 if orig_phenotype == adv_phenotype else 1.0
            
            # Calculate allele drift (Jaccard distance)
            if orig_alleles or adv_alleles:
                intersection = len(orig_alleles & adv_alleles)
                union = len(orig_alleles | adv_alleles)
                allele_drift = 1.0 - (intersection / union) if union > 0 else 1.0
            else:
                allele_drift = 0.0
            
            # Combined drift score
            drift_score = (phenotype_drift + allele_drift) / 2.0
            
            bt.logging.info(f"Drift measurement - Phenotype: {phenotype_drift}, Alleles: {allele_drift}, Combined: {drift_score}")
            return drift_score
            
        except Exception as e:
            bt.logging.error(f"Error measuring drift: {str(e)}")
            return 1.0  # Maximum penalty for errors
    
    def validate_miner_response(self, 
                              vcf_content: str, 
                              non_causal_snps: List[str] = None,
                              pharmcat_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete validation pipeline for a miner response.
        
        Args:
            vcf_content: VCF content from miner
            non_causal_snps: Optional list of non-causal SNPs for adversarial testing
            pharmcat_results: Optional pre-computed PharmaCAT results to avoid redundant computation
            
        Returns:
            Dict containing comprehensive validation results
        """
        try:
            
            # 1. Get ground truth from PharmCAT (use provided results if available)
            if pharmcat_results is None:
                pharmcat_results = self.get_ground_truth(vcf_content)
            else:
                bt.logging.debug(f"Using provided PharmaCAT results (avoiding redundant computation)")
            gold_label = self.extract_gold_label(pharmcat_results)
            
            # 2. Adversarial QC (if non-causal SNPs provided)
            adversarial_results = None
            drift_score = 0.0
            if non_causal_snps:
                adversarial_vcf = self.create_adversarial_vcf(vcf_content, non_causal_snps)
                adversarial_results = self.get_ground_truth(adversarial_vcf)
                drift_score = self.measure_drift(pharmcat_results, adversarial_results)
            
            # 4. Calculate final validation score
            validation_score = self._calculate_validation_score(
                gold_label, drift_score,
            )
            
            validation_results = {
                "validation_score": validation_score,
                "gold_label": gold_label,
                "pharmcat_results": pharmcat_results,
                "adversarial_results": adversarial_results,
                "drift_score": drift_score,
            }
            
            bt.logging.info(f"Validation completed - Score: {validation_score:.4f}")
            return validation_results
            
        except Exception as e:
            bt.logging.error(f"Error in validation pipeline: {str(e)}")
            return {
                "error": str(e),
                "validation_score": 0.0,
                "gold_label": None,
                "pharmcat_results": None,
                "adversarial_results": None,
                "drift_score": 1.0,
                "chi_square_test": {"error": str(e), "p_value": 1.0}
            }
    
    def _calculate_validation_score(self, 
                                   gold_label: Dict[str, Any], 
                                   drift_score: float) -> float:
        """
        Calculate final validation score based on all validation components.
        
        Args:
            gold_label: Gold label from PharmCAT
            drift_score: Drift score from adversarial testing
            chi_square_results: Chi-square test results
            
        Returns:
            Final validation score (0.0 to 1.0)
        """
        try:
            # Base score from gold label quality
            base_score = 1.0 if gold_label.get("phenotype") and "error" not in gold_label else 0.0
            
            # Penalty for high drift (adversarial robustness)
            drift_penalty = drift_score * 0.3  # 30% penalty for high drift
            
            # Calculate final score
            final_score = max(0.0, base_score - drift_penalty)
            
            bt.logging.info(f"Validation score calculation: base={base_score}, drift_penalty={drift_penalty}, final={final_score}")
            return final_score
            
        except Exception as e:
            bt.logging.error(f"Error calculating validation score: {str(e)}")
            return 0.0
