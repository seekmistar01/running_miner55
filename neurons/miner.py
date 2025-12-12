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
import sys
import time

import stdpopsim

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

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

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

        Returns:
            VCF file content as string (with metadata embedded)
        Raises:
            Exception: simulating error

        """

        temp_vcf = None

        try:
            # Extract parameters from JSON schema task with defaults (matching validator defaults)
            population_model = task.get("population_model")
            population = task.get("population")
            genome_model = task.get("genome-model")
            chromosome = task.get("chromosome")

            # Create temporary file for final output
            temp_vcf = tempfile.NamedTemporaryFile(suffix='.vcf', delete=False)
            temp_vcf.close()

            # Try stdpopsim simulation
            if self._simulate_genome(temp_vcf.name, population_model, population, genome_model, chromosome):
                # Read VCF content from file with error handling
                try:
                    with open(temp_vcf.name, 'r') as f:
                        vcf_content = f.read()
                except IOError as e:
                    raise Exception(f"Could not open {temp_vcf.name}: {e}")

                # Add metadata to VCF
                vcf_content = self._add_vcf_metadata(vcf_content, task)

                bt.logging.debug(f"added metadata to {temp_vcf}")

                return vcf_content

        except Exception as e:
            raise Exception(f"VCF generation error for task {task}: {e}")

        finally:
            # Ensure cleanup of temporary files in all cases
            try:
                if os.path.exists(temp_vcf.name):
                    os.unlink(temp_vcf.name)

                ts_file = temp_vcf.name + ".ts"
                if os.path.exists(ts_file):
                    os.unlink(ts_file)

            except Exception as cleanup_error:
                bt.logging.warning(f"Error cleaning up temp files: {cleanup_error}")

    def _simulate_genome(self, vcf_name: str, population_model: str,
                         population: str, genome_model: str, chromosome: Any) -> bool:
        """Run stdpopsim simulation."""
        try:

            # Parse chromosome specification (supports single, multiple, ranges)
            chrom_value = self._parse_chromosome_spec(chromosome)[0]
            chromosome_chr = f"chr{chrom_value}"

            species_name = "HomSap"

            # Get species with caching and specific error handling
            species = stdpopsim.get_species(species_name)

            # Get demographic model with caching and specific error handling
            model = species.get_demographic_model(population_model)

            # Get contig with genetic map with caching and specific error handling
            contig = species.get_contig(chromosome_chr, genetic_map=genome_model, mutation_rate=model.mutation_rate)

            # Get engine with caching and specific error handling
            engine = stdpopsim.get_engine("msprime")

            # Set up our sample for single genome
            samples = {population: 1}

            # Simulate with error handling
            try:
                bt.logging.debug(f"Start simulation")
                ts = engine.simulate(model, contig, samples)
            except Exception as e:
                raise Exception(f"Simulation failed: {e}")

            if ts is None:
                raise Exception("Simulation returned None (no tree sequence generated)")

            # Write to VCF with correct chromosome name
            try:
                bt.logging.debug(f"Write vcf to {vcf_name}")
                with open(vcf_name, "w") as out:
                    ts.write_vcf(out, contig_id=chromosome_chr)
            except IOError as e:
                raise Exception(f"Failed to write VCF file to {vcf_name}: {e}")
            except Exception as e:
                raise Exception(f"Error writing VCF file: {e}")

            # Verify file was created and has content
            if not os.path.exists(vcf_name) or os.path.getsize(vcf_name) == 0:
                raise Exception(f"VCF file was not created or is empty: {vcf_name}")

            bt.logging.debug(f"Simulation finished")
            return True

        except Exception as e:
            raise Exception(f"Unexpected simulation error: {e}")

    def _add_vcf_metadata(self, vcf_content: str, task: Optional[Dict[str, Any]] = None) -> str:
        """
        Add metadata to VCF content including miner identification, and task parameters.

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
