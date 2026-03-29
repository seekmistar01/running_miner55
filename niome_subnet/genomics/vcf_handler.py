import json
import os
import datetime
import re
import time
import requests
from typing import List, Dict, Any
import bittensor as bt
from niome_subnet.utils import upload_file_to_s3
from niome_subnet.genomics.model import ValidationContext, GenomicSimulationTask, MinerScoreDto
from niome_subnet.utils.constants import (
    MINER_SCORE_URL,
    BASE_DELAY_SECONDS,
    MAX_SUBMIT_RETRIES,
    METADATA_SCORE_WEIGHT
)

def save_vcf(vcf_content, validation_context: ValidationContext, task: GenomicSimulationTask, file_name: str):
    """Utility to create a VCF file from a string content."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs('./vcf_files', exist_ok=True)

        # Delete old file
        if len(file_name) > 0:
            output_path = os.path.join("./vcf_files", file_name)
            if os.path.exists(output_path):
                os.remove(output_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")    
        file_name = f"vcf_t{task.get('id')}_{timestamp}_{validation_context.miner_hotkey}_m{validation_context.miner_uid}_{validation_context.validator_hotkey}_v{validation_context.validator_uid}.vcf"
        output_path = os.path.join("./vcf_files", file_name)

        with open(output_path, "w") as vcf_file:
            vcf_file.write(vcf_content)
        
        return file_name
    except Exception as e:
         bt.logging.error(f"Error saving VCF file: {e}")
         raise ValueError(f"Error saving VCF filename '{file_name}': {e}")


def _cleanup_vcf_directory(open_files: List[Any] = []):
    """
    Remove all VCF files from the local vcf_files directory.
    """
    vcf_dir = "./vcf_files"

    if open_files.__len__() > 0:
        for f in open_files:
            try:
                f.close()
            except Exception as e:
                bt.logging.error(f"Error closing file: {e}")

    if not os.path.exists(vcf_dir):
        return

    deleted = 0
    errors = 0

    for filename in os.listdir(vcf_dir):
        if not filename.endswith(".vcf"):
            continue

        file_path = os.path.join(vcf_dir, filename)

        try:
            os.remove(file_path)
            deleted += 1
        except Exception as e:
            errors += 1
            bt.logging.error(f"Failed to delete VCF file {filename}: {e}")


def _parse_vcf_filename(file_name: str) -> tuple[str, str, int]:
    """
    Parse task_id, miner_hotkey, miner_uid from VCF filename.
    Expected format:
    vcf_t{task_id}_{timestamp}_{miner_hotkey}_m{miner_uid}_{validator_hotkey}_v{validator_uid}.vcf
    Returns (task_id, miner_hotkey, miner_uid) or raises ValueError.
    """
    base = os.path.basename(file_name)
    # Match filename format:
    # vcf_t{task_id}_{timestamp}_{miner_hotkey}_m{miner_uid}_{validator_hotkey}_v{validator_uid}.vcf
    # - task_id: characters up to the first underscore (UUIDs with hyphens are allowed)
    # - timestamp: ignored here
    # - miner_hotkey: anything up to the literal `_m{miner_uid}` marker (allows underscores)
    # - miner_uid: digits
    # - validator_hotkey: anything up to `_v{validator_uid}`
    pattern = r"^vcf_t(?P<task_id>[^_]+)_(?P<timestamp>[^_]+_[^_]+)_(?P<miner_hotkey>.+?)_m(?P<miner_uid>\d+)_(?P<validator_hotkey>.+?)_v(?P<validator_uid>\d+)\.vcf$"
    match = re.match(pattern, base)
    if not match:
        raise ValueError(f"Invalid VCF filename format: {file_name}")
    try:
        task_id = match.group("task_id")
        miner_hotkey = match.group("miner_hotkey")
        miner_uid = int(match.group("miner_uid"))
        return task_id, f"{miner_uid}_{miner_hotkey}", miner_uid
    except Exception as e:
        raise ValueError(f"Error parsing VCF filename '{file_name}': {e}")


def get_top3_vcf_files(top_rankings: List[Dict[str, Any]]) -> List[str]:
    """
    Process VCF files: Find and keep top 3, return their paths.

    Args:
        top_rankings: Top 3 ranking entries

    Returns:
        List of paths to top 3 VCF files
    """
    vcf_dir = "./vcf_files"  # Default VCF directory
    top_vcf_files = []

    for ranking in top_rankings:
        uid = ranking["uid"]
        hotkey = ranking["hotkey"]

        vcf_file = ""
        # Find VCF file for this miner
        if not os.path.exists(vcf_dir):
            continue

        # Look for matching files
        for filename in os.listdir(vcf_dir):
            if filename.endswith(".vcf"):
                # Check if filename contains both UID and hotkey patterns
                if hotkey in filename:
                    vcf_file = os.path.join(vcf_dir, filename)

        if vcf_file:
            top_vcf_files.append(vcf_file)

    return top_vcf_files


def submit_validation_result(
    self,
    scores: List[float],
    weights: List[float],
    top3_vcf_files: List[str],  # Only top 3 for file upload, but data from all VCFs
):
    """
    Submit validation results with retry logic, robust filename parsing, and streaming uploads.
    - Parse VCF filenames for task_id and miner_hotkey.
    - Build payload data and files for top 3 VCFs.
    - Use retry logic for backend POST.
    - Use streaming file uploads to minimize memory usage.
    """
    try:
        # Pre-size lists for all possible miner_uids
        open_files = []
        miner_scores = []

        # 1. Build data payload from all VCFs in ./vcf_files
        vcf_dir = "./vcf_files"
        if os.path.exists(vcf_dir):
            for fname in os.listdir(vcf_dir):
                if not fname.endswith(".vcf"):
                    continue
                vcf_path = os.path.join(vcf_dir, fname)

                try:
                    task_id, miner_hotkey, miner_uid = _parse_vcf_filename(vcf_path)
                except Exception as e:
                    bt.logging.warning(f"Invalid VCF filename {vcf_path}: {e}")
                    continue

                if scores[miner_uid] > (METADATA_SCORE_WEIGHT + 0.1):
                    miner_score = MinerScoreDto(
                        task_id=task_id,
                        miner_hotkey=miner_hotkey,
                        score=scores[miner_uid],
                        weight=weights[miner_uid],
                    )
                    miner_scores.append(miner_score)

        # 2. Build files payload from top3_vcf_files only
        for vcf_path in top3_vcf_files:
            try:
                s3_path = upload_file_to_s3(self, vcf_path)
                task_id, miner_hotkey, miner_uid = _parse_vcf_filename(vcf_path)
                for ms in miner_scores:
                    if ms.task_id == task_id and ms.miner_hotkey == miner_hotkey:
                        ms.file_path = s3_path
                        break
            except Exception as e:
                bt.logging.error(f"Error uploading VCF file {vcf_path} to S3: {e}")
                continue

        for attempt in range(1, MAX_SUBMIT_RETRIES + 1):
            try:
                payload = {
                    "miner_scores": [
                        ms.to_dict()
                        for ms in miner_scores
                    ]
                }
                timestamp = str(time.time())
                canonical = json.dumps({
                    'payload': json.dumps(payload, separators=(',', ':'), sort_keys=True),
                    'hotkey': self.wallet.hotkey.ss58_address,
                    'netuid': str(self.netuid),
                    'timestamp': timestamp,
                }, separators=(',', ':'), sort_keys=True)

                signature = self.wallet.hotkey.sign(canonical).hex()

                response = requests.post(
                    MINER_SCORE_URL,
                    headers=self.build_signature_headers(
                        signature=signature,
                        hotkey=self.wallet.hotkey.ss58_address,
                        timestamp=timestamp,
                        netuid=str(self.netuid),
                    ),
                    json=payload
                )
                if response.status_code == 200:
                    bt.logging.info("Validation result submitted successfully")
                    break
                else:
                    bt.logging.error(
                        f"Backend submission failed: {response.status_code} | {response.text}"
                    )
                    if attempt == MAX_SUBMIT_RETRIES:
                        bt.logging.error("All retries failed, giving up.")
            except Exception as e:
                bt.logging.error(f"Error submitting validation result (attempt {attempt}): {e}")
                if attempt == MAX_SUBMIT_RETRIES:
                    bt.logging.error("All retries failed, giving up.")
            if attempt < MAX_SUBMIT_RETRIES:
                delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                bt.logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        _cleanup_vcf_directory(open_files)
    except Exception as e:
        bt.logging.error(f"Unexpected error in submit_validation_result: {e}")
        _cleanup_vcf_directory(open_files)
        raise ValueError(f"Unexpected error in submit_validation_result: {e}")
        
