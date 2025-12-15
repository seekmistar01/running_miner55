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
from typing import List
import bittensor as bt
import numpy as np
import aiohttp

from niome_subnet.protocol import GenomicsTaskSynapse
from niome_subnet.validator.reward import get_rewards
from niome_subnet.utils.uids import get_miner_uids, get_random_uids
from niome_subnet.genomics.model import GenomicSimulationTask
import niome_subnet.utils.constants as config
from niome_subnet.utils.constants import ( BASE_URL)


async def generate_task(self) -> GenomicSimulationTask | dict[str, str | int]:
    """Generate a synthetic genomic simulation task."""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/api/task",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:

                if response.status != 200:
                    raise RuntimeError(
                        f"Backend returned status {response.status}"
                    )

                data = await response.json()

                bt.logging.info("Task successfully fetched from backend")
                return data
            
    except Exception as e:
        bt.logging.error(f"Error on generating task: {e}, returns the sample data")
        raise e


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

        # Adjust the scores based on responses from miners.
        rewards = get_rewards(self, query=self.step, responses=responses, task=task, miner_uids = miner_uids)

        bt.logging.info(f"Scored responses: {rewards}")

        # Update the scores.
        self.update_scores(rewards, miner_uids)
    except Exception as e:
        bt.logging.error(f"Error during forward step: {e}")

    time.sleep(5)
