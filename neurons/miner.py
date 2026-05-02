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

"""
NIOME miner neuron: uses subnet stack (BaseMinerNeuron, GenomicsTaskSynapse) and
reference_free_miner.pipeline for FASTQ -> k-mer / de Bruijn -> VCF generation.
"""

import hashlib
import os
import sys
import time
from typing import Tuple

# Repo root must be on path before niome_subnet / reference_free_miner (works without PYTHONPATH).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import bittensor as bt

from niome_subnet.base.miner import BaseMinerNeuron
from niome_subnet.protocol import GenomicsTaskSynapse
from reference_free_miner.pipeline import generate_vcf_for_task

bt.logging.on()


class Miner(BaseMinerNeuron):
    """
    Miner that answers GenomicsTaskSynapse requests with VCF text produced by the
    reference-free caller (no local reference FASTA; uses task FASTQs and region).
    """

    MAX_RETRIES = 3

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

    async def forward(self, synapse: GenomicsTaskSynapse) -> GenomicsTaskSynapse:
        """
        Run reference-free variant calling on the task FASTQs and fill synapse outputs.
        Uses: read1/read2 paths or URLs, genome_context, expected_variant_count, timeout.
        """
        start_time = time.time()

        try:
            if synapse.task is None:
                raise ValueError("Missing task in synapse")

            task_data = synapse.task.model_dump()
            bt.logging.info(f"Processing genomics task: {task_data}")

            timeout_seconds = float(synapse.timeout or 30.0)

            vcf_content, caller_meta = generate_vcf_for_task(
                synapse.task,
                timeout_seconds=timeout_seconds,
                max_reads_per_file=20000,
            )

            elapsed_time = time.time() - start_time
            vcf_hash = hashlib.sha256(vcf_content.encode()).hexdigest()

            answer_json = {
                "vcf_hash": vcf_hash,
                "vcf_length": len(vcf_content),
                "task_parameters": task_data,
                "model_version": "reference-free-kmer-dbg-v1",
                "caller_meta": caller_meta,
                "timestamp": time.time(),
            }

            synapse.vcf_content = vcf_content
            synapse.elapsed_time = elapsed_time
            synapse.answer_json = answer_json

            bt.logging.info(
                f"Generated VCF: {answer_json['vcf_length']} chars, "
                f"returned={caller_meta.get('returned_count')}, "
                f"fillers={caller_meta.get('filler_count')}, "
                f"time={elapsed_time:.2f}s"
            )
            bt.logging.debug(f"VCF preview: {vcf_content[:500]}")

        except Exception as e:
            bt.logging.error(f"Forward error: {e}")
            elapsed = time.time() - start_time
            synapse.vcf_content = None
            synapse.elapsed_time = elapsed
            synapse.answer_json = {
                "error": str(e),
                "model_version": "reference-free-kmer-dbg-v1",
                "timestamp": time.time(),
            }
            synapse.error = str(e)

        return synapse

    async def blacklist(self, synapse: GenomicsTaskSynapse) -> Tuple[bool, str]:
        """Reject missing identity, unregistered callers, or non-validators if configured."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey "
                    f"{synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: GenomicsTaskSynapse) -> float:
        """Higher metagraph stake -> higher processing priority."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
