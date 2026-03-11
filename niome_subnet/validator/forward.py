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
import json
import time
import uuid
import wandb
from typing import List, Optional

import bittensor as bt
import numpy as np

import niome_subnet.utils.constants as config
from niome_subnet.protocol import GenomicsTaskSynapse
from niome_subnet.validator.reward import get_rewards
from niome_subnet.utils.uids import get_miner_uids, get_random_uids
from niome_subnet.genomics.model import GenomicSimulationTask, ValidationContext, TaskPayload
from niome_subnet.utils.constants import ( TASK_REQUEST_TIMEOUT, BASE_DELAY_SECONDS, MAX_TASK_RETRIES, GET_TASK_URL )

sem = asyncio.Semaphore(config.MINER_QUERY_K)

async def generate_task(self) -> GenomicSimulationTask | dict[str, str | int]:
    """Generate a synthetic genomic simulation task with retry logic and fallback."""
    payload = TaskPayload(
        timestamp=time.time(),
        hotkey=self.wallet.hotkey.ss58_address,
        uuid=str(uuid.uuid4()),
        netuid=str(self.netuid),
    )
    timestamp = str(time.time())
    canonical = json.dumps({
        'payload': json.dumps(payload.model_dump(), separators=(',', ':'), sort_keys=True),
        'hotkey': self.wallet.hotkey.ss58_address,
        'netuid': str(self.netuid),
        'timestamp': timestamp,
    }, separators=(',', ':'), sort_keys=True)

    signature = self.wallet.hotkey.sign(canonical).hex()

    for attempt in range(1, MAX_TASK_RETRIES + 1):
        try:
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    GET_TASK_URL,
                    headers=self.build_signature_headers(
                        signature=signature,
                        hotkey=self.wallet.hotkey.ss58_address,
                        timestamp=timestamp,
                        netuid=str(self.netuid),
                    ),
                    json=payload.model_dump(),
                    timeout=aiohttp.ClientTimeout(total=TASK_REQUEST_TIMEOUT),
                ) as response:
                    if response.status != 201:
                        raise RuntimeError(
                            f"Backend returned status {response.status}"
                        )
                    data = await response.json()
                    return data
        except Exception as e:
            bt.logging.error(f"Error on generating task (attempt {attempt}): {e}")
            if attempt < MAX_TASK_RETRIES:
                delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                bt.logging.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                bt.logging.error("All retries failed, returning fallback sample data")
                raise e

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

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    try:
        if len(self.remain_miner_uids) == 0:
            self.remain_miner_uids = get_miner_uids(self)

        miner_uids = get_random_uids(
            self, k=config.MINER_QUERY_K, available_uids=self.remain_miner_uids
        )

        bt.logging.info(f"Sending task to miners: {miner_uids}")
        self.remain_miner_uids = self.remain_miner_uids[
            ~np.isin(self.remain_miner_uids, miner_uids)
        ]

        task = await generate_task(self)

        bt.logging.info(f"Sending task to miners: {task}")

        synapse = GenomicsTaskSynapse(task=task, timeout=config.FORWARD_TIMEOUT)

        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        # The dendrite client queries the network.
        tasks = [asyncio.create_task(query_axon_limited(self, axon, synapse)) for axon in axons]
        responses: List[Optional[GenomicsTaskSynapse]] = await asyncio.gather(*tasks, return_exceptions=True)

        # Create Validation Context
        validation_contexts = [
            ValidationContext(
                miner_uid=uid,
                miner_hotkey=self.metagraph.hotkeys[uid],
                validator_uid=self.uid,
                validator_hotkey=self.wallet.hotkey.ss58_address,
            )
            for uid in miner_uids
        ]

        # Adjust the scores based on responses from miners.
        rewards = get_rewards(self, query=self.step, responses=responses, task=task, validation_contexts=validation_contexts)

        bt.logging.info(f"Scored responses: {rewards}")

        # Update the scores.
        await self.update_scores(rewards, miner_uids)
    except Exception as e:
        bt.logging.error(f"Error during forward step: {e}")

    time.sleep(5)
