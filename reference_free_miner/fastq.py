from __future__ import annotations

import gzip
import io
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


DNA_BASES = set("ACGT")


@dataclass(slots=True)
class ReadRecord:
    """One FASTQ record after optional quality trimming."""

    name: str
    seq: str
    qual: str

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def mean_q(self) -> float:
        if not self.qual:
            return 0.0
        return sum(phred33(c) for c in self.qual) / len(self.qual)


def phred33(ch: str) -> int:
    """Convert FASTQ Phred+33 quality character to Q score."""
    return max(0, ord(ch) - 33)


def resolve_fastq_path(path_or_url: str, work_dir: str | Path) -> Path:
    """
    Return a local path for a FASTQ path or URL.

    Miner tasks normally give URLs. This downloader is intentionally simple and
    uses stdlib only.
    """
    if path_or_url.startswith(("http://", "https://")):
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        suffix = ".fastq.gz" if path_or_url.endswith(".gz") else ".fastq"
        local = work / ("download_" + str(abs(hash(path_or_url))) + suffix)
        if not local.exists() or local.stat().st_size == 0:
            urllib.request.urlretrieve(path_or_url, local)
        return local
    return Path(path_or_url)


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def iter_fastq(path: str | Path, max_reads: Optional[int] = None) -> Iterator[ReadRecord]:
    """
    Stream FASTQ records.

    FASTQ format:
      line1: @read_id
      line2: sequence
      line3: +
      line4: quality
    """
    count = 0
    with _open_text(Path(path)) as fh:
        while True:
            name = fh.readline()
            if not name:
                break
            seq = fh.readline()
            plus = fh.readline()
            qual = fh.readline()
            if not qual:
                break

            name = name.strip()
            seq = seq.strip().upper()
            plus = plus.strip()
            qual = qual.strip()

            if not name.startswith("@"):
                continue
            if not plus.startswith("+"):
                continue
            if len(seq) != len(qual):
                # malformed read; skip
                continue

            yield ReadRecord(name=name[1:], seq=seq, qual=qual)
            count += 1
            if max_reads is not None and count >= max_reads:
                break


def trim_low_quality(record: ReadRecord, min_base_q: int = 15) -> Optional[ReadRecord]:
    """Trim low-quality bases from both ends."""
    if record.length == 0:
        return None

    qs = [phred33(c) for c in record.qual]
    left = 0
    right = len(qs)

    while left < right and qs[left] < min_base_q:
        left += 1
    while right > left and qs[right - 1] < min_base_q:
        right -= 1

    if right <= left:
        return None

    return ReadRecord(
        name=record.name,
        seq=record.seq[left:right],
        qual=record.qual[left:right],
    )


def clean_reads(
    paths: Iterable[str | Path],
    *,
    max_reads_per_file: int = 20000,
    min_read_len: int = 30,
    min_mean_q: float = 18.0,
    min_base_q: int = 12,
    max_n_fraction: float = 0.05,
) -> list[ReadRecord]:
    """Read and quality-filter FASTQ records."""
    cleaned: list[ReadRecord] = []

    for path in paths:
        for rec in iter_fastq(path, max_reads=max_reads_per_file):
            rec = trim_low_quality(rec, min_base_q=min_base_q)
            if rec is None:
                continue
            if rec.length < min_read_len:
                continue
            if rec.mean_q < min_mean_q:
                continue
            if rec.seq.count("N") / max(rec.length, 1) > max_n_fraction:
                continue
            if any(ch not in "ACGTN" for ch in rec.seq):
                continue
            cleaned.append(rec)

    return cleaned


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]
