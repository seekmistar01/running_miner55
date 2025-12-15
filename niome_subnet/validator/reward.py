# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 genomes.io

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
import numpy as np
from typing import List
from niome_subnet.protocol import GenomicsTaskSynapse
from niome_subnet.genomics.model import GenomicSimulationTask, ValidationContext
from niome_subnet.genomics.validate_response import validate_response

def calculate_score(query: int, response: GenomicsTaskSynapse, task: GenomicSimulationTask, validation_context : ValidationContext) -> float:
    """
    Reward the miner response to the request. This method returns a reward
    value for the miner, which is used to update the miner"s score.

    Returns:
    - float: The reward value for the miner.her
    """
    # Checking miner"s response with task
    score = validate_response(response, task, validation_context)
    return score


def get_rewards(
    self,
    query: int,
    responses: List[GenomicsTaskSynapse],
    task: GenomicSimulationTask,
    miner_uids : List[int]
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[GenomicTaskSynapse]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.        

    validator_uid = getattr(self, 'uid', -1)
    validator_hotkey = getattr(self.wallet.hotkey, 'ss58_address', 'unknown') if hasattr(self, 'wallet') else 'unknown'
    
    rewards = []
    
    for idx, (response, miner_uid) in enumerate(zip(responses, miner_uids)):
        if not response or getattr(response.dendrite, 'status_code', 0) != 200:
            rewards.append(0.0)
            continue
        
        # Get miner hotkey
        miner_hotkey = self.metagraph.hotkeys[miner_uid] if (
            hasattr(self, 'metagraph') and 
            self.metagraph is not None and
            0 <= miner_uid < len(self.metagraph.hotkeys)
        ) else 'unknown'
        
        # Create metadata object
        validation_context = ValidationContext(
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            validator_uid=validator_uid,
            validator_hotkey=validator_hotkey,
        )
        
        # Calculate score
        score = calculate_score(
            query=query,
            response=response,
            task=task,
            validation_context=validation_context
        )
        
        rewards.append(score)
    
    return np.array(rewards)
