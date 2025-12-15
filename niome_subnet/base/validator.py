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


import copy
from niome_subnet.utils.constants import SCORE_EMA_ALPHA
import numpy as np
import asyncio
import argparse
import threading
import os
import requests
from datetime import datetime, timezone
import bittensor as bt

from typing import List, Union, Dict, Any
from traceback import print_exception

from niome_subnet.base.neuron import BaseNeuron
from niome_subnet.base.utils.weight_utils import (
    process_scores,
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
) 
from niome_subnet.mock import MockDendrite
from niome_subnet.utils.config import add_validator_args


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

        self.remain_miner_uids = np.array([])

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")
    
    def build_signed_headers(self) -> dict:
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())
        message = f"<Signature>{timestamp}</Signature>"
        signature = self.wallet.hotkey.sign(message)
        return {
            "X-Validator-Hotkey": self.wallet.hotkey.ss58_address,
            "X-Validator-Signature": signature.hex(),
            "X-Validator-Timestamp": str(timestamp),
        }
    
    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        scores = np.nan_to_num(self.scores)
        original_scores = np.nan_to_num(self.scores).tolist()
        processed_scores = process_scores(scores)
        norm = np.linalg.norm(processed_scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = processed_scores / norm

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Select Top 3 .vcf files
        rankings = self._calculate_rankings(raw_weights)
        top_vcf_files = self._process_vcf_files(rankings[:3])
        self._submit_validation_result(
            scores=original_scores,
            weights=raw_weights.tolist(),
            vcf_files=top_vcf_files
        )

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((self.metagraph.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards= np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        self.scores = SCORE_EMA_ALPHA * scattered_rewards + (1 - SCORE_EMA_ALPHA) * self.scores
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # # Save the state of the validator to file.
        # np.savez(
        #     self.config.neuron.full_path + "/state.npz",
        #     step=self.step,
        #     scores=self.scores,
        #     hotkeys=self.hotkeys,
        # )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # # Load the state of the validator from file.
        # state = np.load(self.config.neuron.full_path + "/state.npz")
        # self.step = state["step"]
        # self.scores = state["scores"]
        # self.hotkeys = state["hotkeys"]

    def _calculate_rankings(self, scores: List[float]) -> List[Dict[str, Any]]:
        """
        Calculate rankings based on scores.
        
        Args:
            scores: List of scores for each miner
        
        Returns:
            List of dictionaries with uid, score, and rank
        """
        # Get UIDs from metagraph
        uids = self.metagraph.uids.tolist()
        
        # Create list of (uid, score) pairs
        uid_score_pairs = list(zip(uids, scores))
        
        # Sort by score in descending order
        sorted_pairs = sorted(uid_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Create rankings with rank information
        rankings = []
        for rank, (uid, score) in enumerate(sorted_pairs, 1):
            rankings.append({
                'uid': int(uid),
                'score': float(score),
                'rank': rank,
                'hotkey': self.metagraph.hotkeys[uid] if uid < len(self.metagraph.hotkeys) else 'unknown'
            })
        
        bt.logging.info(f"Calculated rankings: {rankings}")
        return rankings
    
    def _process_vcf_files(self, top_rankings: List[Dict[str, Any]]) -> List[str]:
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
            uid = ranking['uid']
            hotkey = ranking['hotkey']
            
            # Find VCF file for this miner
            vcf_file = self._find_vcf_for_miner(vcf_dir, uid, hotkey)
            
            if vcf_file:
                top_vcf_files.append(vcf_file)
                bt.logging.info(f"Found VCF for top miner UID {uid}: {vcf_file}")
            else:
                bt.logging.warning(f"No VCF file found for top miner UID {uid}")
        
        return top_vcf_files
    
    def _find_vcf_for_miner(self, vcf_dir: str, miner_uid: int, miner_hotkey: str) -> str:
        """
        Find VCF file for a specific miner.
        
        Args:
            vcf_dir: Directory containing VCF files
            miner_uid: Miner's UID
            miner_hotkey: Miner's hotkey
        
        Returns:
            Path to VCF file if found, empty string otherwise
        """
        if not os.path.exists(vcf_dir):
            return ""
        
        # Look for matching files
        for filename in os.listdir(vcf_dir):
            if filename.endswith('.vcf'):
                # Check if filename contains both UID and hotkey patterns
                if miner_hotkey in filename:
                    return os.path.join(vcf_dir, filename)
        
        # If not found with patterns, try more flexible search
        for filename in os.listdir(vcf_dir):
            if filename.endswith('.vcf') and f"m{miner_uid}" in filename:
                return os.path.join(vcf_dir, filename)
        
        return ""
    
    def _submit_validation_result(
        self,
        scores: List[float],
        weights: List[List[float]],
        vcf_files: List[str],  # TOP-3 ONLY → FILE UPLOAD ONLY
    ):
        """
        Submit validation results.
        - Parse metadata from ALL VCFs in ./vcf_files
        - Upload ONLY top-3 VCF files
        """
        backend_url = "https://your-backend-api.com/submit-validation"
        vcf_dir = "./vcf_files"

        try:
            # -------------------------------
            # 1. Parse ALL VCF metadata
            # -------------------------------
            miner_meta = {}  # miner_uid -> (task_id, miner_hotkey)

            for fname in os.listdir(vcf_dir):
                if not fname.endswith(".vcf"):
                    continue

                path = os.path.join(vcf_dir, fname)

                try:
                    task_id, miner_hotkey, miner_uid = self._parse_vcf_filename(path)
                    miner_meta[miner_uid] = (task_id, miner_hotkey)
                except Exception as e:
                    bt.logging.warning(f"Skipping invalid VCF {fname}: {e}")

            if not miner_meta:
                bt.logging.error("No valid VCF metadata found")
                return

            # -------------------------------
            # 2. Build payload ONLY for TOP-3
            # -------------------------------
            data = []
            files = []
            open_files = []

            for vcf_path in vcf_files:
                if not os.path.exists(vcf_path):
                    bt.logging.warning(f"Missing top VCF: {vcf_path}")
                    continue

                task_id, miner_hotkey, miner_uid = self._parse_vcf_filename(vcf_path)

                if miner_uid not in miner_meta:
                    bt.logging.warning(f"Miner UID {miner_uid} not found in metadata")
                    continue

                score = float(scores[miner_uid])
                weight = float(weights[miner_uid][miner_uid])

                data.extend([
                    ("task_ids", task_id),
                    ("miners", miner_hotkey),
                    ("scores", str(score)),
                    ("weights", str(weight)),
                ])

                f = open(vcf_path, "rb")
                open_files.append(f)
                files.append(
                    ("files", (os.path.basename(vcf_path), f, "text/vcf"))
                )

            if not files:
                bt.logging.error("No top-3 VCFs attached")
                return

            # -------------------------------
            # 3. POST
            # -------------------------------
            response = requests.post(
                backend_url,
                data=data,
                files=files,
                timeout=30,
            )

            if response.status_code == 200:
                bt.logging.info("Validation result submitted successfully")
            else:
                bt.logging.error(
                    f"Backend submission failed: {response.status_code} | {response.text}"
                )

        except Exception as e:
            bt.logging.error(f"Error submitting validation result: {e}")

        finally:
            for f in open_files:
                f.close()
            self._cleanup_vcf_directory()


    def _cleanup_vcf_directory(self):
        """
        Remove all VCF files from the local vcf_files directory.
        """
        vcf_dir = "./vcf_files"

        if not os.path.exists(vcf_dir):
            bt.logging.debug("VCF directory does not exist, nothing to clean")
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

        bt.logging.info(
            f"VCF cleanup finished — deleted={deleted}, errors={errors}"
        )

    def _parse_vcf_filename(self, vcf_path: str) -> tuple[str, str, int]:
        """
        Parse task_id, miner_hotkey, miner_uid from VCF filename.

        Format:
        vcf_t{task_id}_{timestamp}_{miner_hotkey}_m{miner_uid}_{validator_hotkey}_v{validator_uid}.vcf
        """
        filename = os.path.basename(vcf_path)

        parts = filename[:-4].split("_")

        task_id = parts[0][5:]          # vcf_t{task_id}
        miner_hotkey = parts[2]         # {miner_hotkey}
        miner_uid = int(parts[3][1:])   # m{uid}

        return task_id, miner_hotkey, miner_uid
