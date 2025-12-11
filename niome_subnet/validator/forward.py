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

import time
from typing import Dict, Any, List, Coroutine
import bittensor as bt
import numpy as np
import aiohttp

from niome_subnet.protocol import GenomicsTaskSynapse
from niome_subnet.validator.reward import get_rewards
from niome_subnet.utils.uids import get_miner_uids, get_random_uids
from niome_subnet.genomics.model import GenomicSimulationTask

import niome_subnet.utils.constants as config


async def generate_task(self) -> GenomicSimulationTask | dict[str, str | int]:
    """Generate a synthetic genomic simulation task."""

    endpoint = f"{config.GENOMIC_STORAGE_URL}/generate"

    try:
        async with aiohttp.ClientSession() as session:
            header = self.build_signed_headers()
            payload = {}

            response = await session.post(
                endpoint, json=payload, headers=header, timeout=10
            )
            response.raise_for_status()
            return GenomicSimulationTask.model_validate(response.json())

    except Exception as e:
        bt.logging.error(f"Error on generating task: {e}, returns the sample data")

        return {
            "simulator": "stdpopsim",
            "population_model": "OutOfAfrica_4J17",
            "population": "CHB",
            "genome_model": "PyrhoCHB_GRCh38",
            "chromosome": 10,
            "output": "vcf",
        }


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    if len(self.remain_miner_uids) == 0:
        self.remain_miner_uids = get_miner_uids(self)

    miner_uids = get_random_uids(
        self, k=config.MINER_QUERY_K, available_uids=self.remain_miner_uids
    )

    bt.logging.info(f"Sending task to miners: {miner_uids}")
    self.remain_miner_uids = self.remain_miner_uids[
        ~np.isin(self.remain_miner_uids, miner_uids)
    ]

    bt.logging.info(f"Remaning miner uids: {self.remain_miner_uids}")

    task = await generate_task(self)

    bt.logging.info(f"Sending task to miners: {task}")

    synapse = GenomicsTaskSynapse(task=task, timeout=config.FORWARD_TIMEOUT)

    axons = [self.metagraph.axons[uid] for uid in miner_uids]

    # The dendrite client queries the network.
    responses: List[GenomicsTaskSynapse] = await self.dendrite(
        axons=axons, synapse=synapse, deserialize=False, timeout=config.FORWARD_TIMEOUT
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    # TODO(developer): Define how the validator scores responses.
    # Adjust the scores based on responses from miners.
    rewards = get_rewards(self, query=self.step, responses=responses, task=task)

    bt.logging.info(f"Scored responses: {rewards}")

    # Update the scores.
    self.update_scores(rewards, miner_uids)

    time.sleep(5)
