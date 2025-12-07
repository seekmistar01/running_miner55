# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Genomes.io
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt

from typing import Dict, Any, Tuple, Optional, List
import hashlib
import json
import os
import tempfile
import random
import sys
import time
import urllib

import stdpopsim
import vcfpy

# Bittensor Miner Template:
import niome_subnet

# import base miner class which takes care of most of the boilerplate
from niome_subnet.base.miner import BaseMinerNeuron
from niome_subnet.protocol import GenomicsTaskSynapse

bt.logging.on()

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    MAX_RETRIES = 3  # Maximum number of retry attempts for network operations

    # Task schema definitions
    EXPECTED_FIELDS = {
        "simulator": str,  # e.g., "stdpopsim"
        "population_model": str,  # e.g., "OutOfAfrica_4J17"
        "population": str,  # e.g., "CEU", "YRI", "CHB", "JPT"
        "genome-model": str,  # e.g., "PyrhoCEU_GRCh38"
        "chromosome": (int, str, list),
        # Single: 1-22, "1"-"22", "X", "chr1"; Multiple: "1,2,3", ["1","2"]; Range: "1-5", "chr1-chr5"
        "output": str,  # e.g., "vcf"
        "alleles": str,  # Optional: URL or reference
    }
    REQUIRED_FIELDS = ["simulator", "population_model", "population", "chromosome", "output"]
    VALID_POPULATIONS = ["CEU", "YRI", "CHB", "JPT", "AFR", "AMR", "EAS", "EUR", "SAS"]

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Cache for stdpopsim objects to avoid recreating them on every call
        self._species_cache = {}  # Cache for species objects
        self._model_cache = {}  # Cache for demographic model objects (species_name, population_model)
        self._contig_cache = {}  # Cache for contig objects (species_name, chromosome, genome_model)
        self._engine_cache = None  # Cache for engine (usually msprime, same for all)

    async def forward(self, synapse: GenomicsTaskSynapse) -> GenomicsTaskSynapse:
        """
        Processes the incoming 'GenomicsTaskSynapse' synapse by performing a predefined operation on the input data.

        Args:
            synapse (GenomicsTaskSynapse): The synapse object containing the task.

        Returns:
            GenomicsTaskSynapse: The synapse object with the updated filed on the miner's processing logic.

        """

        try:
            # Process genomics task (VCF generation)
            start_time = time.time()

            # dump dict from model
            task_data = synapse.task.model_dump()

            bt.logging.info(f"Processing genomics task: {task_data}")

            # Validate JSON schema task format
            self._validate_task(task_data)

            # Generate VCF file based on JSON schema task
            vcf_content = self._generate_vcf_from_task(task_data)

            if not vcf_content or not isinstance(vcf_content, str) or len(vcf_content) == 0:
                raise Exception("Failed to generate VCF content")

            # Check timeout window
            elapsed_time = time.time() - start_time

            # Create structured answer JSON (use hash instead of full content)
            vcf_hash = hashlib.sha256(vcf_content.encode()).hexdigest()
            answer_json = {
                "vcf_hash": vcf_hash,
                "vcf_length": len(vcf_content),
                "task_parameters": task_data,
                "generation_time": elapsed_time,
                "model_version": "1.0",
                "timestamp": time.time()
            }

            # Generate signature for answer JSON
            answer_str = json.dumps(answer_json, sort_keys=True)
            signature = self._generate_signature(answer_str, 0.0)

            # Return VCF content to validator (as required)
            synapse.vcf_content = vcf_content
            synapse.answer_json = answer_json
            synapse.signature = signature

            bt.logging.info(
                f"Generated VCF file from JSON schema task: {answer_json['vcf_length']} characters, "
                f"signature: {signature[:16]}..., "
                f"time: {elapsed_time:.2f}s"
            )
            bt.logging.debug(f"VCF content preview: {vcf_content[:200]}...")

        except Exception as e:
            bt.logging.error(f"Forward error: {e}")
            synapse.error = str(e)
            return synapse

        return synapse

    def _generate_signature(self, answer_str: str, confidence: float) -> str:
        """Generate cryptographic signature for answer."""
        data = f"{answer_str}:{confidence}:{time.time()}"
        signature = hashlib.sha256(data.encode()).hexdigest()
        return signature

    def _generate_vcf_from_task(self, task: Dict[str, Any]) -> str | None:
        """
        Generate VCF file from JSON schema task parameters.

        Args:
            task: JSON schema task dictionary with fields:
                - simulator: simulation tool (e.g., "stdpopsim")
                - population_model: population model name (e.g., OutOfAfrica_4J17)
                - population: population identifier (CEU, YRI, etc.)
                - genome-model: genome model identifier (e.g., PyrhoCEU_GRCh38)
                - chromosome: chromosome specification (single: "1"-"22", "X", "chr1";
                  multiple: "1,2,3", ["1","2"]; range: "1-5", "chr1-chr5")
                - output: output format ("vcf")
                - alleles: optional alleles reference URL
                - md5_upload_url: optional URL endpoint for HTTP POST of MD5 hash

        Returns:
            VCF file content as string (with metadata embedded)
        Raises:
            Exception: simulating error

        """

        temp_vcf = None
        temp_vcf_sim = None

        try:
            # Extract parameters from JSON schema task with defaults (matching validator defaults)
            population_model = task.get("population_model")
            population = task.get("population")
            genome_model = task.get("genome-model")
            chromosome = task.get("chromosome")

            # Create temporary file for simulation
            temp_vcf_sim = tempfile.NamedTemporaryFile(suffix='.vcf', delete=False)
            temp_vcf_sim.close()

            # Create temporary file for final output
            temp_vcf = tempfile.NamedTemporaryFile(suffix='.vcf', delete=False)
            temp_vcf.close()

            # Parse chromosome specification (supports single, multiple, ranges)
            chrom_value = self._parse_chromosome_spec(chromosome)[0]
            chromosome_chr = f"chr{chrom_value}"

            # get allele_definition_file path and ensure file exists and is valid
            allele_definition_file = self._get_allele_file_path(task)

            if not allele_definition_file:
                raise Exception("CYP2C9 allele definition CSV not available")

            # Try stdpopsim simulation
            if self._simulate_genome(temp_vcf_sim.name, population_model, population, genome_model, chromosome_chr):

                # iterate vcf with allele_definition_file and write to temp_vcf
                allele_definitions, reference_calls = self._read_allele_definitions(allele_definition_file)

                if 'allele_one' in task:
                    allele_one = task.get('allele_one')
                else:
                    allele_one = self._pick_random_allele(allele_definitions)
                    bt.logging.info(f"randomly picked {allele_one} as first allele")

                if 'allele_two' in task:
                    allele_two = task.get('allele_two')
                else:
                    allele_two = self._pick_random_allele(allele_definitions)
                    bt.logging.info(f"randomly picked {allele_two} as second allele")

                self._iterate_vcf(
                    temp_vcf_sim.name, temp_vcf.name,
                    allele_definitions, reference_calls, allele_one, allele_two, chromosome_chr
                )

                # Read VCF content from file with error handling
                try:
                    with open(temp_vcf.name, 'r') as f:
                        vcf_content = f.read()
                except IOError as e:
                    raise Exception(f"Could not open {temp_vcf.name}: {e}")

                # Add metadata to VCF (block hash, miner hotkey, task parameters)
                vcf_content = self._add_vcf_metadata(vcf_content, task)

                return vcf_content

        except Exception as e:
            # Include task context for better debugging
            task_context = {
                "population_model": task.get("population_model", "unknown"),
                "population": task.get("population", "unknown"),
                "genome_model": task.get("genome-model", "unknown"),
                "chromosome": task.get("chromosome", "unknown")
            }
            raise Exception(f"VCF generation error for task {task_context}: {e}")
        finally:
            # Ensure cleanup of temporary files in all cases
            try:
                if os.path.exists(temp_vcf.name):
                    os.unlink(temp_vcf.name)

                if os.path.exists(temp_vcf_sim.name):
                    os.unlink(temp_vcf_sim.name)

                ts_file = temp_vcf_sim.name + ".ts"
                if os.path.exists(ts_file):
                    os.unlink(ts_file)
            except Exception as cleanup_error:
                bt.logging.warning(f"Error cleaning up temp files: {cleanup_error}")

    def _simulate_genome(self, vcf_name: str, population_model: str,
                         population: str, genome_model: str, chromosome_chr: str) -> bool:
        """Run stdpopsim simulation."""
        try:
            species_name = "HomSap"

            # Get species with caching and specific error handling
            try:
                if species_name not in self._species_cache:
                    self._species_cache[species_name] = stdpopsim.get_species(species_name)
                    bt.logging.debug(f"Cached species: {species_name}")
                species = self._species_cache[species_name]
            except (KeyError, ValueError) as e:
                bt.logging.error(f"Failed to get species '{species_name}': {e}")
                return False
            except Exception as e:
                bt.logging.error(f"Unexpected error getting species '{species_name}': {e}")
                return False

            # Get demographic model with caching and specific error handling
            model_cache_key = f"{species_name}_{population_model}"
            try:
                if model_cache_key not in self._model_cache:
                    model = species.get_demographic_model(population_model)
                    self._model_cache[model_cache_key] = model
                    bt.logging.debug(f"Cached demographic model: {model_cache_key}")
                else:
                    model = self._model_cache[model_cache_key]
            except KeyError as e:
                # Try to list available models for better error message
                available_models = list(species.list_demographic_models())
                raise Exception(f"Demographic model '{population_model}' not found. "
                                f"Available models: {available_models}")
            except Exception as e:
                raise Exception(f"Error getting demographic model '{population_model}': {e}")

            # Get contig with genetic map with caching and specific error handling
            contig_cache_key = f"{species_name}_{chromosome_chr}_{genome_model}_{model.mutation_rate}"
            try:
                if contig_cache_key not in self._contig_cache:
                    contig = species.get_contig(chromosome_chr, genetic_map=genome_model,
                                                mutation_rate=model.mutation_rate)
                    self._contig_cache[contig_cache_key] = contig
                    bt.logging.debug(f"Cached contig: {contig_cache_key}")
                else:
                    contig = self._contig_cache[contig_cache_key]
            except KeyError as e:
                raise Exception(f"Contig '{chromosome_chr}' or genetic map '{genome_model}' not found: {e}")
            except Exception as e:
                raise Exception(f"Error getting contig '{chromosome_chr}' with genetic map '{genome_model}': {e}")

            # Get engine with caching and specific error handling
            try:
                if self._engine_cache is None:
                    self._engine_cache = stdpopsim.get_engine("msprime")
                    bt.logging.debug("Cached msprime engine")
                engine = self._engine_cache
            except (KeyError, ValueError) as e:
                raise Exception(f"Failed to get msprime engine: {e}")
            except Exception as e:
                raise Exception(f"Unexpected error getting engine: {e}")

            # Validate population exists in model before simulation
            try:
                # Try to get available populations from model
                if hasattr(model, 'populations'):
                    available_populations = {p.name: p for p in model.populations}
                    if population not in available_populations:
                        raise Exception(
                            f"Population '{population}' not found in demographic model '{population_model}'. "
                            f"Available populations: {list(available_populations)}")
                else:
                    # Some models might not expose populations list - log warning but continue
                    bt.logging.debug(
                        f"Could not validate population '{population}' - model doesn't expose populations list")
            except Exception as pop_validation_error:
                # If validation fails, log warning but continue (some models might not support this)
                bt.logging.warning(f"Could not validate population '{population}': {pop_validation_error}")

            # Set up our sample for single genome
            samples = {population: 1}

            # Simulate with error handling
            try:
                ts = engine.simulate(model, contig, samples)
            except Exception as e:
                raise Exception(f"Simulation failed: {e}")

            if ts is None:
                raise Exception("Simulation returned None (no tree sequence generated)")

            # Write to VCF with correct chromosome name
            try:
                with open(vcf_name, "w") as out:
                    ts.write_vcf(out, contig_id=chromosome_chr)
            except IOError as e:
                raise Exception(f"Failed to write VCF file to {vcf_name}: {e}")
            except Exception as e:
                raise Exception(f"Error writing VCF file: {e}")

            # Verify file was created and has content
            if not os.path.exists(vcf_name) or os.path.getsize(vcf_name) == 0:
                raise Exception(f"VCF file was not created or is empty: {vcf_name}")

            return True

        except Exception as e:
            raise Exception(f"Unexpected simulation error: {e}")

    def _add_vcf_metadata(self, vcf_content: str, task: Optional[Dict[str, Any]] = None) -> str:
        """
        Add metadata to VCF content including block hash, miner identification, and task parameters.

        Args:
            vcf_content: VCF file content as string
            task: Optional task dictionary with task parameters

        Returns:
            VCF content with metadata embedded in headers
        """
        try:
            # Validate VCF content is not empty
            if not vcf_content or not vcf_content.strip():
                bt.logging.warning("VCF content is empty, cannot add metadata")
                return vcf_content

            # Optimize for large VCFs: find header section and insertion point efficiently
            # VCF format: headers start with ##, data starts with #CHROM, then data lines
            # We only need to process headers, not the entire file

            # Find the end of headers (where #CHROM line is - this is the column header)
            # Look for the line that starts with #CHROM (can be at start or after newline)
            chrom_header_pos = vcf_content.find('#CHROM\t')
            if chrom_header_pos == -1:
                # Try without tab (some VCFs use spaces)
                chrom_header_pos = vcf_content.find('#CHROM')

            if chrom_header_pos != -1:
                # Find the start of the #CHROM line (go back to previous newline or start)
                line_start = vcf_content.rfind('\n', 0, chrom_header_pos)
                if line_start == -1:
                    line_start = 0
                else:
                    line_start += 1  # Skip the newline itself

                # Header section is everything before the #CHROM line
                header_section = vcf_content[:line_start]
                # Data section is #CHROM line and everything after
                data_section = vcf_content[line_start:]
            else:
                # No #CHROM line found, treat entire content as headers
                header_section = vcf_content
                data_section = ""

            header_lines = header_section.split('\n')

            # Find where to insert metadata (after fileformat, before other headers)
            insert_index = 1  # After ##fileformat line
            for i, line in enumerate(header_lines):
                if line.startswith('##fileformat'):
                    insert_index = i + 1
                    break

            # Build metadata lines
            metadata_lines = []

            # Add Bittensor block and miner identification (always add these)
            try:
                current_block = self.block
                metadata_lines.append(f"##bittensor_block={current_block}\n")
            except Exception as block_error:
                bt.logging.warning(f"Could not get current block: {block_error}")

            try:
                hotkey_address = self.wallet.hotkey.ss58_address
                metadata_lines.append(f"##miner_hotkey={hotkey_address}\n")

                # Also add a hash of the hotkey for shorter identification
                hotkey_hash = hashlib.sha256(hotkey_address.encode()).hexdigest()[:16]
                metadata_lines.append(f"##miner_hotkey_hash={hotkey_hash}\n")
            except Exception as hotkey_error:
                bt.logging.warning(f"Could not get hotkey: {hotkey_error}")

            # Add task properties if provided
            if task:
                if 'population_model' in task:
                    metadata_lines.append(f"##population_model={task['population_model']}\n")
                if 'population' in task:
                    metadata_lines.append(f"##population={task['population']}\n")
                if 'genome-model' in task:
                    metadata_lines.append(f"##genome_model={task['genome-model']}\n")
                if 'chromosome' in task:
                    metadata_lines.append(f"##chromosome={task['chromosome']}\n")
                if 'allele_one' in task:
                    metadata_lines.append(f"##allele_one={task['allele_one']}\n")
                if 'allele_two' in task:
                    metadata_lines.append(f"##allele_two={task['allele_two']}\n")

            # Insert metadata lines into header
            header_lines[insert_index:insert_index] = metadata_lines

            # Reconstruct: header with metadata + data section
            # Join header lines and append data section (avoiding double newline)
            header_with_metadata = '\n'.join(header_lines)
            if data_section:
                # Ensure proper newline between header and data
                if not header_with_metadata.endswith('\n'):
                    header_with_metadata += '\n'
                return header_with_metadata + data_section
            else:
                return header_with_metadata

        except Exception as metadata_error:
            bt.logging.warning(f"Could not add metadata to VCF: {metadata_error}")
            return vcf_content  # Return original content if metadata addition fails

    def _validate_task(self, task: Dict[str, Any]) -> None:
        """
        Validate JSON schema task against expected format.
        Uses cached class constants for schema definitions to avoid recreation overhead.

        Args:
            task: Task dictionary from validator
        Raises:
            Exception: Task schema validation error
        """
        errors = []

        if not isinstance(task, dict):
            raise Exception(f"Invalid task format: expected dict, got {type(task)}")

        # Check for required fields (using cached class constant)
        for field in self.REQUIRED_FIELDS:
            if field not in task:
                errors.append(f"Missing required field: {field}")

        # Validate field types (using cached class constant)
        for field, expected_type in self.EXPECTED_FIELDS.items():
            if field in task:
                # Special handling for chromosome field (can be int, str, or list)
                if field == "chromosome":
                    if not isinstance(task[field], (int, str, list)):
                        errors.append(
                            f"Field '{field}' has wrong type: expected int, str, or list, got {type(task[field])}")
                elif not isinstance(task[field], expected_type):
                    if not (isinstance(expected_type, tuple) and isinstance(task[field], expected_type)):
                        errors.append(
                            f"Field '{field}' has wrong type: expected {expected_type}, got {type(task[field])}")

        # Validate chromosome is valid (supports single, multiple, and ranges)
        if "chromosome" in task:
            parsed_chroms = self._parse_chromosome_spec(task["chromosome"])
            if not parsed_chroms:
                errors.append(f"Invalid chromosome specification: {task['chromosome']}")

        # Validate population is one of expected values (using cached class constant)
        if "population" in task:
            if task["population"] not in self.VALID_POPULATIONS:
                bt.logging.warning(f"Unknown population: {task['population']}")

        if errors:
            raise Exception(f"Task schema validation failed: {', '.join(errors)}")

    def _parse_chromosome_spec(self, chromosome_spec: Any) -> List[str]:
        """
        Parse chromosome specification supporting single, multiple, and range formats.

        Supports:
        - Single: "1", "chr1", "X", 1
        - Multiple: "1,2,3", ["1", "2", "3"], "chr1,chr2,chrX"
        - Range: "1-5", "chr1-chr5", "1-22"
        - Mixed: "1-3,5,X" or "chr1-chr3,chr5,chrX"

        Args:
            chromosome_spec: Chromosome specification (int, str, or list)

        Returns:
            List of normalized chromosome strings (without "chr" prefix)
        """
        valid_chroms = [str(i) for i in range(1, 23)] + ["X", "Y", "MT", "M"]
        chromosomes = []

        # Handle list input
        if isinstance(chromosome_spec, list):
            for chrom in chromosome_spec:
                chromosomes.extend(self._parse_chromosome_spec(chrom))
            return chromosomes

        # Convert to string and normalize
        chrom_str = str(chromosome_spec).strip()
        if not chrom_str:
            return []

        # Remove "chr" prefix for processing
        chrom_str = chrom_str.replace("chr", "")

        # Check for comma-separated values (multiple chromosomes)
        if "," in chrom_str:
            parts = [p.strip() for p in chrom_str.split(",")]
            for part in parts:
                chromosomes.extend(self._parse_chromosome_spec(part))
            return chromosomes

        # Check for range (e.g., "1-5" or "chr1-chr5")
        if "-" in chrom_str:
            parts = chrom_str.split("-", 1)
            if len(parts) != 2:
                return []

            start_str = parts[0].strip()
            end_str = parts[1].strip()

            # Try to parse as numeric range
            try:
                start = int(start_str)
                end = int(end_str)

                # Validate range
                if start < 1 or end > 22 or start > end:
                    return []

                # Generate range
                return [str(i) for i in range(start, end + 1)]
            except ValueError:
                # Not a numeric range, treat as invalid
                return []

        # Single chromosome - validate and return
        if chrom_str in valid_chroms:
            return [chrom_str]

        return []

    def _calculate_file_md5(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash as hexadecimal string
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            bt.logging.error(f"Error calculating MD5 for {file_path}: {e}")
            return ""

    def _validate_allele_file_structure(self, csv_path: str) -> bool:
        """
        Validate the structure of the allele definition CSV file.
        Checks that all rows have consistent column counts matching the header.

        This validation is performed when downloading new files to catch malformed data early.

        Args:
            csv_path: Path to the CSV file

        Returns:
            True if file structure is valid, False otherwise
        """
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                lines = csvfile.readlines()

            if not lines:
                bt.logging.error(f"Allele definition file is empty: {csv_path}")
                return False

            # Parse header
            header_line = lines[0].strip('\n')
            positions = header_line.split("\t")[1:]  # skip first field (allele name)
            num_positions = len(positions)

            if num_positions == 0:
                bt.logging.error(f"Allele definition file has no positions in header: {csv_path}")
                return False

            bt.logging.debug(f"Allele definition file header has {num_positions} positions")

            # Validate each data row
            errors = []
            for i, line in enumerate(lines[1:], start=2):  # Start at line 2 (skip header)
                tokens = line.strip('\n').split("\t")

                if len(tokens) < 2:
                    errors.append(f"Row {i}: Too few columns (expected at least 2, got {len(tokens)})")
                    continue

                allele_name = tokens[0]
                allele_definition = tokens[1:]
                num_variants = len(allele_definition)

                # Check bounds: number of variants should match number of positions
                if num_variants > num_positions:
                    errors.append(
                        f"Row {i} (allele '{allele_name}'): Has {num_variants} variants "
                        f"but header only has {num_positions} positions. "
                        f"Extra variants will be ignored."
                    )
                elif num_variants < num_positions:
                    errors.append(
                        f"Row {i} (allele '{allele_name}'): Has {num_variants} variants "
                        f"but header has {num_positions} positions. "
                        f"Missing variants will use reference calls."
                    )

            if errors:
                bt.logging.error(
                    f"Allele definition file structure validation failed for {csv_path}:"
                )
                for error in errors[:10]:  # Show first 10 errors
                    bt.logging.error(f"  - {error}")
                if len(errors) > 10:
                    bt.logging.error(f"  ... and {len(errors) - 10} more errors")

                # Reject file with structural errors
                return False

            bt.logging.info(f"Allele definition file structure validation passed: {csv_path}")
            return True

        except Exception as e:
            bt.logging.error(f"Error validating allele file structure for {csv_path}: {e}")
            return False

    def _verify_file_integrity(self, csv_path: str, md5_path: str) -> bool:
        """
        Verify file integrity by comparing calculated MD5 with stored MD5.

        Args:
            csv_path: Path to the CSV file
            md5_path: Path to the MD5 file

        Returns:
            True if file exists and MD5 matches, False otherwise
        """
        if not os.path.exists(csv_path):
            return False

        if not os.path.exists(md5_path):
            bt.logging.warning(f"MD5 file not found at {md5_path}")
            return False

        try:
            # Read expected MD5 from file
            with open(md5_path, 'r') as f:
                expected_md5 = f.read().strip().split()[0]  # Handle format like "hash filename"

            # Calculate actual MD5
            actual_md5 = self._calculate_file_md5(csv_path)

            if not actual_md5:
                return False

            # Compare MD5s
            if actual_md5.lower() == expected_md5.lower():
                bt.logging.info(f"MD5 verification passed for {csv_path}")
                return True
            else:
                bt.logging.warning(f"MD5 mismatch for {csv_path}: expected {expected_md5}, got {actual_md5}")
                return False

        except Exception as e:
            bt.logging.error(f"Error verifying file integrity: {e}")
            return False

    def _download_file(self, url: str, file_path: str, max_retries: int = None) -> bool:
        """
        Download a file from URL to local path with retry logic.

        Args:
            url: URL to download from
            file_path: Local path to save the file
            max_retries: Maximum number of retry attempts (default: MAX_RETRIES)

        Returns:
            True if download successful, False otherwise
        """
        # Use class constant if not specified
        if max_retries is None:
            max_retries = self.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                bt.logging.info(f"Downloading {url} to {file_path} (attempt {attempt + 1}/{max_retries})")
                urllib.request.urlretrieve(url, file_path)
                bt.logging.info(f"Successfully downloaded {file_path}")
                return True
            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, etc.
                    bt.logging.warning(
                        f"Download attempt {attempt + 1} failed: {e}, "
                        f"retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    bt.logging.error(f"Error downloading {url} after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    bt.logging.warning(
                        f"Unexpected error on attempt {attempt + 1}: {e}, "
                        f"retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    bt.logging.error(f"Unexpected error downloading {url} after {max_retries} attempts: {e}")
                    return False

        return False

    def _get_allele_file_path(self, task_data: Dict) -> Optional[str]:
        """
        Get file path and ensure the allele definition file exists and is valid.
        Checks for file and MD5, verifies integrity, and downloads if needed.

        Returns:
            Path to the CSV file if available, None otherwise
        """
        # Use the data directory relative to project root
        data_dir = os.path.join(PROJECT_ROOT, "data")

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        csv_filename = "allele_file.cached.csv"

        csv_path = os.path.join(data_dir, csv_filename)
        md5_path = os.path.join(data_dir, f"{csv_filename}.md5")

        csv_url = task_data['alleles']
        md5_url = task_data['alleles_md5']

        # Download MD5 file first
        if not self._download_file(md5_url, md5_path):
            bt.logging.error("Failed to download MD5 file")
            return None

        # Check if both files exist and verify integrity
        if os.path.exists(csv_path) and os.path.exists(md5_path):
            if self._verify_file_integrity(csv_path, md5_path):
                bt.logging.info(f"Using existing allele definition file: {csv_path}")
                return csv_path
            else:
                bt.logging.warning("MD5 verification failed, will re-download files")
                # Remove invalid files
                try:
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                except Exception as e:
                    bt.logging.warning(f"Error removing invalid files: {e}")

        # Download files if missing or invalid
        bt.logging.info("Downloading allele definition files...")

        # Download CSV file
        if not self._download_file(csv_url, csv_path):
            bt.logging.error("Failed to download CSV file")
            return None

        # Verify integrity of downloaded file
        if not self._verify_file_integrity(csv_path, md5_path):
            bt.logging.error("Downloaded file failed MD5 verification")
            return None

        # Validate file structure (only for newly downloaded files)
        if not self._validate_allele_file_structure(csv_path):
            bt.logging.error("Downloaded file failed structure validation")
            # Remove invalid file
            try:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                if os.path.exists(md5_path):
                    os.remove(md5_path)
            except Exception as e:
                bt.logging.warning(f"Error removing invalid files: {e}")
            return None

        bt.logging.info(f"Successfully downloaded and verified allele definition file: {csv_path}")
        return csv_path

    def _read_allele_definitions(self, allele_def_file):
        """
        reads a slightly cleaned up CSV file of allele definitions from PharmGKB
        and outputs the allele definitions alongside default values, so
        that the alleles can be injected into simulated VCF files

        Returns the allele definitions and reference calls as tuple
        """
        allele_definitions = {}
        reference_calls = []

        for i, line in enumerate(open(allele_def_file, 'r')):
            if i == 0:
                positions = line.strip('\n').split("\t")[1:]
            if i > 1:  # ignore location & RSID header
                single_allele = line.strip("\n").split("\t")
                allele_name = single_allele[0]
                allele_definition = single_allele[1:]
                allele_definitions[allele_name] = {}
                for j, variant in enumerate(allele_definition):
                    if variant != '':
                        allele_definitions[allele_name][positions[j]] = variant
                    else:
                        allele_definitions[allele_name][positions[j]] = reference_calls[positions[j]]
                if i == 2:
                    reference_calls = allele_definitions[allele_name]
        return (allele_definitions, reference_calls)

    def _construct_record(self, reference_calls, allele_definitions,
                          allele_one, allele_two, allele_pos,
                          chromosome):
        """
        construct VCF record to inject into file based on alleles, pass:
        1. reference calls
        2. full allele definitions
        3. name of allele 1
        4. name of allele 2
        5. position of allele
        6. the chromosome (in 'chrX' format)

        Returns a single VCF record/row with the corresponding two alleles requested
        """
        # start with substitution creation, get ref variant and both calls from alleles
        ref_allele = reference_calls[allele_pos]
        alt_one = allele_definitions[allele_one][allele_pos]
        alt_two = allele_definitions[allele_two][allele_pos]
        # print(ref_allele, alt_one, alt_two)
        # check if we need to make one or two alt alleles (mostly will be one)
        # create list of alt alleles
        if len(set([ref_allele, alt_one, alt_two])) == 1:
            # print('both are ref')
            alt = [vcfpy.Substitution('SNV', random.choice('ATCG'))]
            gt = '0|0'
        elif len(set([ref_allele, alt_one, alt_two])) == 2:
            # print('got only one alt allele')
            if alt_one == ref_allele:
                alt = [vcfpy.Substitution('SNV', alt_two)]
                gt = '0|1'
            else:
                alt = [vcfpy.Substitution('SNV', alt_one)]
                gt = '1|0'
        else:
            # print('got two alt alleles')
            alt = [vcfpy.Substitution('SNV', alt_one), vcfpy.Substitution('SNV', alt_two)]
            gt = '1|2'

        # create genotype call
        call = vcfpy.Call('tsk_0', {'GT': gt})

        record = vcfpy.Record(chromosome,
                              int(allele_pos),
                              [],
                              ref_allele,
                              alt,
                              None, ["PASS"],
                              {},
                              ["GT"],
                              [call])
        return record

    def _iterate_vcf(self, infile, outfile, allele_definitions,
                     reference_calls, allele_one, allele_two, chromosome):
        """
        open a VCF file, iterate over it, replace/add the variants as needed
        and then write output to new file
        """

        reader = vcfpy.Reader.from_path(infile)

        # Build and print header
        # print(reader.header)
        # print(reader.header.samples.names)
        contig_length = list(reader.header.get_lines('contig'))[0].length
        reader.header.add_contig_line({'ID': chromosome, 'length': int(contig_length)})

        # if simulation randomly generated variants for positions of interest,
        # we want to skip these to replace them with our pharma markers
        skip_positions = list(reference_calls.keys())

        writer = vcfpy.Writer.from_path(outfile, reader.header)

        # write all the existing records, except where they conflict with our target allele
        for record in reader:
            if record.POS not in skip_positions:
                record.CHROM = chromosome
                record.ID = "."
                writer.write_record(record)

        # write our own alleles
        for entry in reference_calls:
            new_record = self._construct_record(reference_calls,
                                                allele_definitions,
                                                allele_one,
                                                allele_two,
                                                entry,
                                                chromosome)
            writer.write_record(new_record)

    def _pick_random_allele(self, allele_definitions):
        return random.choice(list(allele_definitions.keys()))

    async def blacklist(self, synapse: niome_subnet.protocol.GenomicsTaskSynapse) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (GenomicsTaskSynapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
                not self.config.blacklist.allow_non_registered
                and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: niome_subnet.protocol.GenomicsTaskSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (GenomicsTaskSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
