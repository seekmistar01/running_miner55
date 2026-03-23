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

import bittensor as bt
import shutil
import tempfile
import pharmcat_runner
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from niome_subnet.utils import run_cmd
from niome_subnet.utils.constants import DOCKER_IMAGE, DOCKER_TIMEOUT, WITH_DOCKER

class PharmCATValidator:
    """
    PharmCAT-based validation system for pharmacogenomic VCF analysis.
    
    This class implements:
    1. Ground-truth retrieval using PharmCAT
    2. Adversarial QC with SNP swapping
    """

    docker_use_sudo: bool = False  # Class variable to track if sudo is needed for Docker
    
    def __init__(self):
        """
        Initialize PharmCAT validator
        """
        if WITH_DOCKER:
            try:
                self._assert_docker_available()
            except Exception as e:
                bt.logging.warning(f"Failed to initialize pharmcat docker: {e}\nPharmCAT validation will use fallback mode")
    
    def _assert_docker_available(self) -> None:
        rc, out, err = run_cmd(["docker", "--version"], timeout=20)
        if rc != 0:
            raise RuntimeError(f"Docker not available: {err or out}")

        # also verify daemon connectivity
        rc, out, err = run_cmd(["docker", "ps"], timeout=20)
        if rc != 0:
            # If permission denied to the docker socket, try a non-interactive sudo check
            err_txt = (err or out or "").lower()
            if "permission denied" in err_txt or "connect: permission denied" in err_txt:
                # try non-interactive sudo; this will fail immediately if a password is required
                rc_sudo, _, _ = run_cmd(["sudo", "-n", "docker", "ps"], timeout=20)
                if rc_sudo == 0:
                    self.docker_use_sudo = True
                    return
                # fallthrough: advise user how to fix permissions
                raise RuntimeError(
                    "Docker daemon socket permission denied. Either run this script with sudo, or add your user to the 'docker' group and re-login.\n"
                    "To test without changing groups, try: 'sudo python3 main.py'\n"
                    "Original error: " + (err or out or "")
                )
            raise RuntimeError(f"Docker daemon not reachable: {err or out}")
    
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
            if WITH_DOCKER:
                return self._run_pharmcat_docker_pipeline(vcf_path)
            else:
                return self._run_pharmcat_local(vcf_path)
                
        except Exception as e:
            bt.logging.error(f"Error running PharmCAT: {str(e)}")
            return {"error": str(e), "match": {}, "phenotype": {}}
    
    def _run_pharmcat_docker_pipeline(
        self,
        vcf_path: str,
        docker_image: str = DOCKER_IMAGE,
        timeout: int = DOCKER_TIMEOUT,
    ) -> Dict[str, Any]:
        
        self._assert_docker_available()

        with tempfile.TemporaryDirectory() as out_dir:
            self._run_pharmcat_with_fallbacks(
                image=docker_image,
                vcf_path=vcf_path,
                out_dir=out_dir,
                timeout=timeout,
            )
            result = self._parse_pharmcat_docker_outputs(out_dir)
            shutil.rmtree(out_dir, ignore_errors=True)
            return result

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
            bt.logging.error(f"Error parsing PharmaCAT files: {e}")
        
        return {
            "match": match_data,
            "phenotype": phenotype_data,
        }
    
    def validate_miner_response(self, 
                              report_content: Dict[str, Any], 
                              chr_drugs: dict[str, dict]) -> float:
        """
        Complete validation pipeline for a miner response.
        
        Args:
            result_content: pharmcat result content (dict)
            pharmcat_results: Optional pre-computed PharmaCAT results to avoid redundant computation
            
        Returns:
            Validation score
        """
        try:
            score = 0
            genes = report_content.get("genes", "")
            
            if isinstance(genes, dict):
                gene_drugs: dict[str, list[str]] = {}
                for chr in list(chr_drugs.keys()):
                    chr_info = chr_drugs[chr]
                    gene = chr_info.get("gene", "")
                    gene_drugs[gene] = chr_info.get("drugs", [])
                
                for gene in genes:
                    gene_info = genes.get(gene)
                    if gene_info.get("phased", "false") == "false":
                        continue
                    elif gene not in list(gene_drugs.keys()):
                        score -= 1
                        continue

                    score += 1

            return max(0, score) / len(gene_drugs)
        except Exception as e:
            bt.logging.error(f"Error in validation pipeline: {str(e)}")
            return 0
    
    def _run_pharmcat_with_fallbacks(
        self,
        image: str,
        vcf_path: str,
        out_dir: str,
        timeout: int,
    ) -> Tuple[str, str]:
        """
        Tries multiple command shapes until one works.
        Returns (stdout, stderr) of the successful run.
        Raises RuntimeError with detailed diagnostics if all fail.
        """
        vcf_path = str(Path(vcf_path).resolve())
        out_dir_path = Path(out_dir).resolve()
        out_dir_path.mkdir(parents=True, exist_ok=True)

        container_out = "/out"

        host_dir = str(Path(vcf_path).resolve().parent)
        container_vcf = f"/data/{Path(vcf_path).name}"

        base = [
            "docker", "run", "--rm",
            "-v", f"{host_dir}:/data",
            "-v", f"{str(out_dir_path)}:{container_out}",
            image,
        ]

        candidates: List[List[str]] = [
            # Try the common CLI shapes. Order matters: prefer the executable
            # observed to work in local testing (`pharmcat_pipeline`).
            # Shell-wrapped forms often match the image entrypoint expectations
            ["/bin/bash", "-lc", f"pharmcat_pipeline -reporterJson --output {container_out} {container_vcf}"],
            ["/bin/bash", "-lc", f"pharmcat_pipeline -reporterJson -o {container_out} {container_vcf}"],
            ["/bin/bash", "-lc", f"pharmcat_pipeline -reporterJson {container_vcf} -o {container_out}"],
            ["pharmcat_pipeline", "-reporterJson", "--output", container_out, container_vcf],
            ["pharmcat_pipeline", "-reporterJson", container_vcf, "--output", container_out],
            ["pharmcat_pipeline", "-reporterJson", "-o", container_out, container_vcf],
            ["pharmcat_pipeline", "-reporterJson", container_vcf, "-o", container_out],
            # Most common: executable "pharmcat" with subcommand "pipeline"
            ["pharmcat", "pipeline", "-reporterJson", "--vcf", container_vcf, "--out", container_out],
            ["pharmcat", "pipeline", "-reporterJson", "--input", container_vcf, "--output", container_out],
            # Some images may expose "pipeline" directly
            ["pipeline", "-reporterJson", "--vcf", container_vcf, "--out", container_out],
            ["pipeline", "-reporterJson", "--input", container_vcf, "--output", container_out],
        ]

        jar = self._find_pharmcat_jar(image, timeout=min(timeout, 120))
        if jar:
            # try invoking the jar with positional input and -o output
            candidates.append(["java", "-jar", jar, "-o", container_out, container_vcf])
            candidates.append(["java", "-jar", jar, "--output", container_out, container_vcf])
            candidates.append(["java", "-jar", jar, container_vcf, "-o", container_out])
            # shell-wrapped jar invocation
            candidates.append(["/bin/bash", "-lc", f"java -jar {jar} -o {container_out} {container_vcf}"])

        attempts: List[Dict[str, Any]] = []

        for args in candidates:
            cmd = base + args
            rc, out, err = run_cmd(cmd, timeout=timeout)
            attempts.append({"cmd": cmd, "rc": rc, "stderr_tail": (err or "")[-1200:], "stdout_tail": (out or "")[-800:]})
            if rc == 0:
                return out, err

        # As a last diagnostic, show entrypoint/cmd
        _, out_i, err_i = run_cmd(
            ["docker", "inspect", image, "--format", "Entrypoint={{json .Config.Entrypoint}} Cmd={{json .Config.Cmd}}"],
            timeout=30,
        )
        inspect_line = (out_i or err_i or "").strip()

        msg = [
            "Unable to run PharmCAT in the Docker image with known command patterns.",
            f"Image: {image}",
            f"docker inspect: {inspect_line}",
            "",
            "Tried the following docker run commands (showing rc and stderr tail):",
        ]
        for a in attempts:
            msg.append(f"- rc={a['rc']} cmd={' '.join(a['cmd'])}")
            if a["stderr_tail"]:
                msg.append(f"  stderr: {a['stderr_tail']}")
            if a.get("stdout_tail"):
                msg.append(f"  stdout: {a['stdout_tail']}")
        raise RuntimeError("\n".join(msg))
    
    def _parse_pharmcat_docker_outputs(self, output_dir: str) -> Dict[str, Any]:
        outp = Path(output_dir)
        json_files = list(outp.rglob("*.report.json"))

        if len(json_files) == 0:
            bt.logging.error(f"No JSON report found in PharmCAT output directory: {output_dir}")
            return {}
        
        report: Dict[str, Any] = json.loads(json_files[0].read_text())
        if isinstance(report, dict):
            return report
        
        return {}
    
    def _find_pharmcat_jar(self, image: str, timeout: int) -> Optional[str]:
        # Look for a jar that smells like PharmCAT.
        rc, out, err = self._docker_sh(
            image,
            r'find / -maxdepth 6 -type f \( -iname "*pharmcat*.jar" -o -iname "PharmCAT*.jar" \) 2>/dev/null | head -n 1',
            timeout=timeout,
        )
        if rc == 0:
            jar = (out or "").strip()
            return jar or None
        return None

    def _docker_sh(self, image: str, sh_cmd: str, timeout: int) -> Tuple[int, str, str]:
        return run_cmd(
            ["docker", "run", "--rm", "--entrypoint", "sh", image, "-lc", sh_cmd],
            timeout=timeout,
        )
