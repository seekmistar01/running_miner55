import bittensor as bt
import math
import numpy as np
from typing import Dict, Tuple, List, Union, Any
from niome_subnet.utils.constants import (
    SCORE_DISTRIBUTION,
    TOP_MIN_ALPHA,
    TOP_MIN_ALPHA_SCALE,
    TOP_MINER_COUNT,
    METADATA_SCORE_WEIGHT
)

U32_MAX = 4294967295
U16_MAX = 65535


def normalize_max_weight(x: np.ndarray, limit: float = 0.1) -> np.ndarray:
    r"""Normalizes the numpy array x so that sum(x) = 1 and the max value is not greater than the limit.
    Args:
        x (:obj:`np.ndarray`):
            Array to be max_value normalized.
        limit: float:
            Max value after normalization.
    Returns:
        y (:obj:`np.ndarray`):
            Normalized x array.
    """
    epsilon = 1e-7  # For numerical stability after normalization

    weights = x.copy()
    values = np.sort(weights)

    if x.sum() == 0 or len(x) * limit <= 1:
        return np.ones_like(x) / x.size
    else:
        estimation = values / values.sum()

        if estimation.max() <= limit:
            return weights / weights.sum()

        # Find the cumulative sum and sorted array
        cumsum = np.cumsum(estimation, 0)

        # Determine the index of cutoff
        estimation_sum = np.array(
            [(len(values) - i - 1) * estimation[i] for i in range(len(values))]
        )
        n_values = (estimation / (estimation_sum + cumsum + epsilon) < limit).sum()

        # Determine the cutoff based on the index
        cutoff_scale = (limit * cumsum[n_values - 1] - epsilon) / (
            1 - (limit * (len(estimation) - n_values))
        )
        cutoff = cutoff_scale * values.sum()

        # Applying the cutoff
        weights[weights > cutoff] = cutoff

        y = weights / weights.sum()

        return y


def convert_weights_and_uids_for_emit(
    uids: np.ndarray, weights: np.ndarray
) -> Tuple[List[int], List[int]]:
    r"""Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.
    Args:
        uids (:obj:`np.ndarray,`):
            Array of uids as destinations for passed weights.
        weights (:obj:`np.ndarray,`):
            Array of weights.
    Returns:
        weight_uids (List[int]):
            Uids as a list.
        weight_vals (List[int]):
            Weights as a list.
    """
    # Checks.
    uids = np.asarray(uids)
    weights = np.asarray(weights)

    if np.min(weights) < 0:
        raise ValueError(
            "Passed weight is negative cannot exist on chain {}".format(weights)
        )
    if np.min(uids) < 0:
        raise ValueError("Passed uid is negative cannot exist on chain {}".format(uids))
    if len(uids) != len(weights):
        raise ValueError(
            "Passed weights and uids must have the same length, got {} and {}".format(
                len(uids), len(weights)
            )
        )
    if np.sum(weights) == 0:
        bt.logging.debug("nothing to set on chain")
        return [], []  # Nothing to set on chain.
    else:
        max_weight = float(np.max(weights))
        weights = [
            float(value) / max_weight for value in weights
        ]  # max-upscale values (max_weight = 1).

    weight_vals = []
    weight_uids = []
    for i, (weight_i, uid_i) in enumerate(list(zip(weights, uids))):
        uint16_val = round(
            float(weight_i) * int(U16_MAX)
        )  # convert to int representation.

        # Filter zeros
        if uint16_val != 0:  # Filter zeros
            weight_vals.append(uint16_val)
            weight_uids.append(uid_i)
    return weight_uids, weight_vals


def process_weights_for_netuid(
    uids,
    weights: np.ndarray,
    netuid: int,
    owner_uid: int,
    subtensor: "bt.Subtensor",
    metagraph: "bt.metagraph" = None,
    exclude_quantile: int = 0,
) -> Union[
    tuple[
        np.ndarray[Any, np.dtype[Any]],
        Union[
            Union[
                np.ndarray[Any, np.dtype[np.floating[Any]]],
                np.ndarray[Any, np.dtype[np.complexfloating[Any, Any]]],
            ],
            Any,
        ],
    ],
    tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray],
    tuple[Any, np.ndarray],
]:
    # Get latest metagraph from chain if metagraph is None.
    if metagraph is None:
        metagraph = subtensor.metagraph(netuid)

    # Cast weights to floats.
    if not isinstance(weights, np.ndarray) or weights.dtype != np.float32:
        weights = weights.astype(np.float32)

    # Network configuration parameters from subtensor.
    quantile = exclude_quantile / U16_MAX
    min_allowed_weights = subtensor.min_allowed_weights(netuid=netuid)
    max_weight_limit = subtensor.max_weight_limit(netuid=netuid)

    # Find all non-zero weights.
    non_zero_weight_idx = np.argwhere(weights > 0).squeeze()
    non_zero_weight_idx = np.atleast_1d(non_zero_weight_idx)
    non_zero_weight_uids = uids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]

    if non_zero_weights.size == 0:
        # Create a weight array of zeros for all miners
        all_weights = np.zeros(metagraph.n, dtype=np.float32)
        # Find the index of owner_uid in the uids array
        owner_indices = np.where(uids == owner_uid)[0]
        if len(owner_indices) == 0:
            bt.logging.error(
                f"owner_uid {owner_uid} not found in uids. "
                "Falling back to uniform distribution."
            )
            final_weights = np.ones(metagraph.n) / metagraph.n
            return np.arange(len(final_weights)), final_weights
        owner_idx = owner_indices[0]
        # Give raw weight 1.0 to the owner (will be normalized to max_weight_limit)
        all_weights[owner_idx] = 1.0
        # Normalize to respect max_weight_limit
        normalized_weights = normalize_max_weight(x=all_weights, limit=max_weight_limit)
        # Find which UIDs now have non-zero weights (should be only owner_uid)
        final_non_zero = np.where(normalized_weights > 0)[0]
        final_uids = uids[final_non_zero]
        final_weights = normalized_weights[final_non_zero]
        return final_uids, final_weights

    if metagraph.n < min_allowed_weights:
        bt.logging.warning(
            "Metagraph size smaller than min_allowed_weights, returning all ones."
        )
        final_weights = np.ones(metagraph.n) / metagraph.n
        return np.arange(len(final_weights)), final_weights

    if non_zero_weights.size < min_allowed_weights:
        bt.logging.warning(
            "Number of non-zero weights less than min_allowed_weights, "
            "creating minimal weights for all."
        )
        weights = np.ones(metagraph.n) * 1e-5
        weights[non_zero_weight_idx] += non_zero_weights
        bt.logging.debug("final_weights", weights)
        normalized_weights = normalize_max_weight(x=weights, limit=max_weight_limit)
        return np.arange(len(normalized_weights)), normalized_weights

    # Compute the exclude quantile and find the weights in the lowest quantile
    max_exclude = max(0, len(non_zero_weights) - min_allowed_weights) / len(
        non_zero_weights
    )
    exclude_quantile = min([quantile, max_exclude])
    lowest_quantile = np.quantile(non_zero_weights, exclude_quantile)

    # Exclude all weights below the allowed quantile.
    non_zero_weight_uids = non_zero_weight_uids[lowest_quantile <= non_zero_weights]
    non_zero_weights = non_zero_weights[lowest_quantile <= non_zero_weights]

    # Normalize weights and return.
    normalized_weights = normalize_max_weight(
        x=non_zero_weights, limit=max_weight_limit
    )
    bt.logging.debug("final_weights", normalized_weights)

    return non_zero_weight_uids, normalized_weights


def process_scores(scores: np.ndarray) -> np.ndarray:
    """
    Convert raw miner scores into a final normalized weight vector:
    - Top {TOP_MINER_COUNT} miners receive most revenue according to SCORE_DISTRIBUTION.
    - Remaining miners share the rest using a linear decreasing distribution.
    - Output always sums to exactly 1.0.
    """
    # Filter positive scores and their indices
    positive_marks = scores > 0
    positive_score_num = np.sum(positive_marks)
    if positive_score_num == 0:
        return np.zeros_like(scores)

    # Sort indices by descending scores
    sorted_indices = np.argsort(-scores)

    # compute alpha
    scale = TOP_MIN_ALPHA_SCALE
    if positive_score_num <= TOP_MINER_COUNT:
        alpha = 1.0
    else:
        arg = (positive_score_num - TOP_MINER_COUNT) / scale
        alpha = 1 - (1 - TOP_MIN_ALPHA) * math.tanh(arg)

    bt.logging.info(f"Computed alpha: {alpha}")

    # Top ratios (sum to 1.0)
    ratios = np.array(list(SCORE_DISTRIBUTION.values()))
    bt.logging.info(f"SCORE_DISTRIBUTION ratios: {ratios}")

    weights = np.zeros_like(scores, dtype=np.float32)

    k = min(TOP_MINER_COUNT, positive_score_num)
    top_ratios = ratios[:k]
    top_sum = np.sum(top_ratios)
    # If positive_score_num < TOP_MINER_COUNT, normalize top_ratios, else just scale
    # Use np.divide with where to avoid division by zero
    norm_top_ratios = np.divide(top_ratios, top_sum, out=np.zeros_like(top_ratios), where=top_sum!=0)
    top_weights = np.where(
        positive_score_num < TOP_MINER_COUNT,
        norm_top_ratios * alpha,
        top_ratios * alpha
    )

    for i in range(k):
        weights[sorted_indices[i]] = top_weights[i]

    if positive_score_num > TOP_MINER_COUNT:
        rest_alloc = 1.0 - alpha
        rest_indices = sorted_indices[10:positive_score_num]
        rest_score = scores[rest_indices]
        if len(rest_score) > 0:
            temp = 1
            softmax = np.exp(rest_score / temp) / np.sum(np.exp(rest_score / temp))
            rest_weights = softmax * rest_alloc
            for i, idx in enumerate(rest_indices):
                weights[idx] = rest_weights[i]

    return weights


def calculate_rankings(mg: Dict[str, Any], scores: List[float], owner_uid: int) -> List[Dict[str, Any]]:
    """
    Calculate rankings based on scores.

    Args:
        scores: List of scores for each miner

    Returns:
        List of dictionaries with uid, score, and rank
    """
    # Get UIDs from metagraph
    uids = mg.uids.tolist()

    # Create list of (uid, score) pairs
    uid_score_pairs = list(zip(uids, scores))

    # Sort by score in descending order
    sorted_pairs = sorted(uid_score_pairs, key=lambda x: x[1], reverse=True)

    # Create rankings with rank information
    rankings = []
    for rank, (uid, score) in enumerate(sorted_pairs, 1):
        if score > (METADATA_SCORE_WEIGHT + 0.1) and uid != owner_uid:
            rankings.append(
                {
                    "uid": int(uid),
                    "score": float(score),
                    "rank": rank,
                    "hotkey": (mg.hotkeys[uid] if uid < len(mg.hotkeys) else "unknown"),
                }
            )

    return rankings
