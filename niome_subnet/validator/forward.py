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

import aiohttp
import asyncio
import bittensor as bt
import copy
import json
import numpy as np
import niome_subnet.utils.constants as config
import os
import time
import urllib.request

from typing import List, Optional
from niome_subnet.genomics.model import GroundTruth, Task, MinerSubmission
from niome_subnet.genomics.scoring import create_mapping_file, score
from niome_subnet.protocol import GenomicsTaskSynapse
from niome_subnet.utils import get_miner_uids, get_random_uids

from niome_subnet.utils.constants import BASE_BLOCK_NUMBER, INTERVAL_BLOCKS

sem = asyncio.Semaphore(config.MINER_QUERY_K)


async def fetch_task(self) -> tuple[Task, GroundTruth]:
    """Generate a synthetic genomic simulation task with retry logic and fallback."""
    payload = {}
    timestamp = str(time.time())
    canonical = json.dumps({
        'payload': '{}',
        'hotkey': self.wallet.hotkey.ss58_address,
        'netuid': str(self.netuid),
        'timestamp': timestamp,
    }, separators=(',', ':'), sort_keys=True)

    signature = self.wallet.hotkey.sign(canonical).hex()

    for attempt in range(1, config.MAX_TASK_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    config.TASK_URL,
                    headers=self.build_signature_headers(
                        signature=signature,
                        hotkey=self.wallet.hotkey.ss58_address,
                        timestamp=timestamp,
                        netuid=str(self.netuid),
                    ),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.TASK_REQUEST_TIMEOUT),
                ) as response:
                    if response.status != 201:
                        raise RuntimeError(
                            f"Backend returned status {response.status}"
                        )

                    data = await response.json()
                    
                    task_url = data.get("task_url", "")
                    ground_truth_url = data.get("ground_truth_url", "")

                    if not task_url or not ground_truth_url:
                        raise RuntimeError("Invalid response from backend")

                    task = await fetch_task_by_url(task_url)
                    ground_truth = await fetch_ground_truth_by_url(ground_truth_url)

                    return task, ground_truth
        except Exception as e:
            bt.logging.error(f"Error on generating task (attempt {attempt}): {e}")
            if attempt < config.MAX_TASK_RETRIES:
                delay = config.BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                bt.logging.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                bt.logging.error("All retries failed, returning fallback sample data")
                raise e

async def fetch_task_by_url(task_url: str) -> Task:
    """Fetch task details from the given URL."""
    try:
        def _fetch():
            with urllib.request.urlopen(task_url, timeout=config.TASK_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read())
        data = await asyncio.to_thread(_fetch)
        return Task(**data)
    except Exception as e:
        bt.logging.error(f"Error fetching task details from {task_url}: {type(e).__name__}: {e}")
        raise

async def fetch_ground_truth_by_url(ground_truth_url: str) -> GroundTruth:
    """Fetch ground truth data from the given URL."""
    try:
        def _fetch():
            with urllib.request.urlopen(ground_truth_url, timeout=config.TASK_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read())
        data = await asyncio.to_thread(_fetch)
        return GroundTruth(**data)
    except Exception as e:
        bt.logging.error(f"Error fetching ground truth from {ground_truth_url}: {type(e).__name__}: {e}")
        raise

async def query_axon(self, axon, synapse) -> Optional[GenomicsTaskSynapse]:
    """Query a single axon and return the response."""
    try:
        start_time = time.perf_counter()
        response: GenomicsTaskSynapse = await self.dendrite.forward(
            axons=axon, synapse=synapse, deserialize=False, timeout=config.FORWARD_TIMEOUT
        )
        if response is not None:
            response.elapsed_time = time.perf_counter() - start_time
            return response
    except Exception as e:
        bt.logging.error(f"Error querying axon {axon}: {e}")
        return None

async def query_axon_limited(self, axon, synapse) -> Optional[GenomicsTaskSynapse]:
    """Query an axon with a semaphore to limit concurrency."""
    async with sem:
        return await query_axon(self, axon, synapse)

async def run_validation(self):
    bt.logging.info("Starting validation process...")
    try:
        os.makedirs("data", exist_ok=True)
        miner_uids = get_miner_uids(self)
        final_scores = []
        miner_task, ground_truth = await fetch_task(self)
        bt.logging.info(f"Fetched task: {miner_task.model_dump()}")
        task = copy.deepcopy(miner_task)

        # Download ground truth data first (ref needed by create_mapping_file)
        urllib.request.urlretrieve(ground_truth.truth_vcf, "data/truth.vcf")
        ground_truth.truth_vcf = "data/truth.vcf"
        urllib.request.urlretrieve(ground_truth.ref, "data/ref.fa")
        ground_truth.ref = "data/ref.fa"

        # Download task reads
        urllib.request.urlretrieve(task.input.read1_fastq, "data/read_1.fq")
        task.input.read1_fastq = "data/read_1.fq"
        urllib.request.urlretrieve(task.input.read2_fastq, "data/read_2.fq")
        task.input.read2_fastq = "data/read_2.fq"

        bam = create_mapping_file(ground_truth.ref, task.input.read1_fastq, task.input.read2_fastq)

        while len(miner_uids) > 0:
            selected_uids = get_random_uids(
                self, k=config.MINER_QUERY_K, available_uids=miner_uids
            )

            bt.logging.info(f"Sending task to miners: {selected_uids}")
            miner_uids = miner_uids[
                ~np.isin(miner_uids, selected_uids)
            ]

            synapse = GenomicsTaskSynapse(task=miner_task, timeout=config.FORWARD_TIMEOUT)

            axons = [self.metagraph.axons[uid] for uid in selected_uids]

            # The dendrite client queries the network.
            tasks = [asyncio.create_task(query_axon_limited(self, axon, synapse)) for axon in axons]
            raw_responses: List[Optional[GenomicsTaskSynapse]] = await asyncio.gather(*tasks, return_exceptions=True)

            responses = [
                (uid, resp)
                for uid, resp in zip(selected_uids, raw_responses)
                if isinstance(resp, GenomicsTaskSynapse) and resp.vcf_content
            ]

            scores = [
                score(
                    MinerSubmission(
                        uid=int(uid),
                        vcf_content=resp.vcf_content,
                        response_time=resp.elapsed_time,
                    ), ground_truth, bam)
                for uid, resp in responses
            ]

            for miner_score in scores:
                if miner_score.final_score > 0:
                    final_scores.append(miner_score)
        
        scores = [(s.uid, s.final_score) for s in final_scores]
        bt.logging.info(f"Scores: {scores}")
        self.set_weights(final_scores, task.task_id)
    except Exception as e:
        bt.logging.error(f"Error during validation process: {e}")
    finally:
        self.is_validating = False

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    try:
        if (self.block - BASE_BLOCK_NUMBER) % INTERVAL_BLOCKS < 5 and not self.is_validating:
            self.is_validating = True
            asyncio.create_task(run_validation(self))
    except Exception as e:
        bt.logging.error(f"Error during forward step: {e}")

    time.sleep(5)
