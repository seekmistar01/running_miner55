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
from typing import List

import numpy as np
import bittensor as bt

from niome_subnet.protocol import GenomicsTaskSynapse
from niome_subnet.genomics.model import GenomicSimulationTask, ValidationContext
from niome_subnet.genomics.vcf_check import is_vcf_valid
from niome_subnet.genomics.metadata_scoring import compute_metadata_score
from niome_subnet.genomics.pharmcat_scoring import compute_pharmcat_score
from niome_subnet.genomics.vcf_handler import save_vcf
from niome_subnet.utils.constants import PHARMCAT_SCORE_WEIGHT, METADATA_SCORE_WEIGHT

def calculate_score(query: int, response: GenomicsTaskSynapse, task: GenomicSimulationTask, validation_context : ValidationContext) -> float:
    """
    Reward the miner response to the request. This method returns a reward
    value for the miner, which is used to update the miner"s score.

    Returns:
    - float: The reward value for the miner.her
    """
    vcf_content = response.vcf_content

    if vcf_content is None:
        bt.logging.error("No VCF content in miner response.")
        return 0.0
    
    if not is_vcf_valid(vcf_content):
        bt.logging.error("Invalid VCF content in miner response.")
        return 0.0
    
    metadata_score = compute_metadata_score(vcf_content, task)
    pharmcat_score = compute_pharmcat_score(vcf_content)
    final_score = (PHARMCAT_SCORE_WEIGHT * pharmcat_score) + (
        METADATA_SCORE_WEIGHT * metadata_score
    )
    save_vcf(vcf_content, validation_context)

    # Checking miner"s response with task
    # score = validate_response(response, task, validation_context)
    return  final_score


def get_rewards(
    self,
    query: int,
    responses: List[GenomicsTaskSynapse],
    task: GenomicSimulationTask,
    validation_contexts : List[ValidationContext],
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
    
    return np.array([calculate_score(query, response, task, validation_context) for response, validation_context in zip(responses, validation_contexts)])

