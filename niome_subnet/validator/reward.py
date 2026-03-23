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
import bittensor as bt
import numpy as np
from typing import List, Optional
from niome_subnet.protocol import GenomicsTaskSynapse
from niome_subnet.genomics.model import GenomicSimulationTask, ValidationContext
from niome_subnet.genomics.vcf_check import is_vcf_valid
from niome_subnet.genomics.metadata_scoring import compute_metadata_score
from niome_subnet.genomics.pharmcat_scoring import compute_pharmcat_score
from niome_subnet.genomics.vcf_handler import save_vcf
from niome_subnet.utils.constants import PHARMCAT_SCORE_WEIGHT, METADATA_SCORE_WEIGHT, ELAPSED_TIME_WEIGHT, RESPONSE_TIMEOUT

def calculate_score(response: GenomicsTaskSynapse, task: GenomicSimulationTask, validation_context : ValidationContext, file_name: str) -> tuple[float, str]:
    """
    Reward the miner response to the request. This method returns a reward
    value for the miner, which is used to update the miner"s score.

    Returns:
    - float: The reward value for the miner.her
    """
    try:
        vcf_content = response.vcf_content

        if vcf_content is None or not is_vcf_valid(vcf_content):
            return 0.0, ""
        
        metadata_score = compute_metadata_score(vcf_content, task)
        pharmcat_score = compute_pharmcat_score(vcf_content, task)
        final_score = (PHARMCAT_SCORE_WEIGHT * pharmcat_score) + (
            METADATA_SCORE_WEIGHT * metadata_score
        )
        time_score = compute_elapsed_time_score(response.elapsed_time, final_score)
        final_score += time_score * ELAPSED_TIME_WEIGHT

        file_name = save_vcf(vcf_content, validation_context, task, file_name)

        # Checking miner"s response with task
        # score = validate_response(response, task, validation_context)
        return final_score, file_name
    except Exception as e:
        bt.logging.error(f"Error calculating score: {e}")
        return 0.0, ""


def compute_elapsed_time_score(elapsed_time: float, vcf_score: float) -> float:
    normalized_time = elapsed_time / RESPONSE_TIMEOUT
    v_score = vcf_score / (1 - ELAPSED_TIME_WEIGHT)
    time_score = (v_score ** 2) * (1 - normalized_time ** 2)
    return time_score


def get_rewards(
    self,
    responses: List[Optional[GenomicsTaskSynapse]],
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
    # Return a Python list of (score, filename) tuples so callers can unzip via zip(*rewards).
    return [
        calculate_score(
            response, task, validation_context, self.file_names[validation_context.miner_uid]
        ) if response is not None else (0.0, "")
        for response, validation_context in zip(responses, validation_contexts)
    ]