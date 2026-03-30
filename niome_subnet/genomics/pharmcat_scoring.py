import tempfile
import bittensor as bt
from pathlib import Path
from niome_subnet.genomics.pharmcat_validator import PharmCATValidator

def compute_pharmcat_score(miner_vcf: str, task: dict) -> float:
    try:
        validator = PharmCATValidator()

        with tempfile.TemporaryDirectory() as temp_dir:
            vcf_path = Path(temp_dir) / "temp.vcf"
            with open(vcf_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(miner_vcf)
            report_content = validator._run_pharmcat(vcf_path)

        return validator.validate_miner_response(
            report_content, task.get("gene_drugs", {})
        )
    except Exception as e:
        bt.logging.error(f"Error computing PharmCAT score: {e}")
        return 0.0
