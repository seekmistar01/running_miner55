import os
import time
import json
import requests
import bittensor as bt
from concurrent.futures import ThreadPoolExecutor, as_completed
from .constants import CHUNK_SIZE, S3_UPLOAD_URL, MAX_CONCURRENT_UPLOADS, MAX_CHUNK_UPLOAD_RETRIES

def upload_file_to_s3(self, file_path: str) -> str:
    filename = os.path.basename(file_path)

    payload={
        'filename': filename,
        'content_type': 'application/octet-stream',
        'total_parts': (os.path.getsize(file_path) + CHUNK_SIZE - 1) // CHUNK_SIZE,
    }
    timestamp = str(time.time())
    canonical = json.dumps({
        'payload': json.dumps(payload, separators=(',', ':'), sort_keys=True),
        'hotkey': self.wallet.hotkey.ss58_address,
        'netuid': str(self.netuid),
        'timestamp': timestamp,
    }, separators=(',', ':'), sort_keys=True)
    signature = self.wallet.hotkey.sign(canonical).hex()

    resp = requests.post(
        f'{S3_UPLOAD_URL}/start-upload',
        headers=self.build_signature_headers(
            signature=signature,
            hotkey=self.wallet.hotkey.ss58_address,
            timestamp=timestamp,
            netuid=str(self.netuid),
        ),
        json=payload
    ).json()

    bt.logging.debug(f"Upload initiation response: {resp}")
    upload_id = resp['upload_id']
    s3_key = resp['s3_key']
    presigned_urls = resp['presigned_urls']

    parts = []

    # First pass: read chunks into memory and map to presigned URLs provided by start-upload.
    part_blobs = []  # list of tuples (part_number, chunk_bytes, presigned_url)
    with open(file_path, 'rb') as f:
        part_number = 1
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            # `presigned_urls` is expected to be a list returned by start-upload; index directly.
            presigned_url = presigned_urls[part_number - 1]
            if not presigned_url:
                bt.logging.error(f"Missing presigned URL for part {part_number}; presigned_urls={presigned_urls}")
                raise RuntimeError(f"Failed to obtain presigned url for part {part_number}")

            part_blobs.append((part_number, chunk, presigned_url))
            part_number += 1

    # Upload parts in parallel with retries
    def _upload_part(part_tuple):
        pn, chunk_bytes, url = part_tuple
        attempt = 0
        backoff = 1
        while attempt < MAX_CHUNK_UPLOAD_RETRIES:
            try:
                resp = requests.put(url, data=chunk_bytes, timeout=60)
                bt.logging.debug(f"Chunk upload response headers (part {pn}): {resp.headers}")
                # ETag may be in different cases
                etag = resp.headers.get('ETag') or resp.headers.get('etag') or resp.headers.get('Etag')
                if etag:
                    # preserve quotes as S3 returns them; CompleteMultipartUpload expects quoted ETags
                    return {'ETag': etag, 'PartNumber': pn}
                else:
                    bt.logging.warning(f"No ETag in response for part {pn}, status {resp.status_code}, body: {resp.text}")
                    # If success but no ETag, still accept 200/201
                    if resp.status_code in (200, 201):
                        return {'ETag': '', 'PartNumber': pn}
                    raise RuntimeError(f"No ETag returned for part {pn}")
            except Exception as e:
                attempt += 1
                bt.logging.warning(f"Upload part {pn} attempt {attempt} failed: {e}")
                time.sleep(backoff)
                backoff *= 2

        bt.logging.error(f"Failed to upload part {pn} after {MAX_CHUNK_UPLOAD_RETRIES} attempts")

        payload={
            'upload_id': upload_id,
            's3_key': s3_key,
        }
        timestamp = str(time.time())
        canonical = json.dumps({
            'payload': json.dumps(payload, separators=(',', ':'), sort_keys=True),
            'hotkey': self.wallet.hotkey.ss58_address,
            'netuid': str(self.netuid),
            'timestamp': timestamp,
        }, separators=(',', ':'), sort_keys=True)
        signature = self.wallet.hotkey.sign(canonical).hex()
        requests.post(
            f'{S3_UPLOAD_URL}/abort-upload',
            headers=self.build_signature_headers(
                signature=signature,
                hotkey=self.wallet.hotkey.ss58_address,
                timestamp=timestamp,
                netuid=str(self.netuid),
            ),
            json=payload
        )

        raise RuntimeError(f"Failed to upload part {pn} after {MAX_CHUNK_UPLOAD_RETRIES} attempts")

    max_workers = min(MAX_CONCURRENT_UPLOADS, len(part_blobs) or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_upload_part, p): p[0] for p in part_blobs}
        for fut in as_completed(futures):
            pn = futures[fut]
            try:
                result = fut.result()
                parts.append(result)
            except Exception as e:
                bt.logging.error(f"Failed to upload part {pn}: {e}")
                raise

        payload={
            'upload_id': upload_id,
            's3_key': s3_key,
            'parts': parts,
        }
        timestamp = str(time.time())
        canonical = json.dumps({
            'payload': json.dumps(payload, separators=(',', ':'), sort_keys=True),
            'hotkey': self.wallet.hotkey.ss58_address,
            'netuid': str(self.netuid),
            'timestamp': timestamp,
        }, separators=(',', ':'), sort_keys=True)
        signature = self.wallet.hotkey.sign(canonical).hex()

        requests.post(
            f'{S3_UPLOAD_URL}/complete-upload',
            headers=self.build_signature_headers(
                signature=signature,
                hotkey=self.wallet.hotkey.ss58_address,
                timestamp=timestamp,
                netuid=str(self.netuid),
            ),
            json=payload
        )

    return s3_key