from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .fastq import ReadRecord, phred33


@dataclass(slots=True)
class KmerStats:
    count: int
    q_sum: int

    @property
    def mean_q(self) -> float:
        return self.q_sum / max(self.count, 1)


def build_kmer_table(
    reads: Iterable[ReadRecord],
    *,
    k: int = 31,
    min_kmer_q: int = 15,
) -> dict[str, KmerStats]:
    """
    Count k-mers with a simple quality weight.

    We skip k-mers containing N and k-mers whose average base quality is too low.
    """
    counts: Counter[str] = Counter()
    q_sums: defaultdict[str, int] = defaultdict(int)

    for read in reads:
        seq = read.seq
        qual = read.qual
        if len(seq) < k:
            continue

        qs = [phred33(c) for c in qual]
        for i in range(0, len(seq) - k + 1):
            kmer = seq[i : i + k]
            if "N" in kmer:
                continue
            q_avg = sum(qs[i : i + k]) // k
            if q_avg < min_kmer_q:
                continue
            counts[kmer] += 1
            q_sums[kmer] += q_avg

    return {kmer: KmerStats(count=c, q_sum=q_sums[kmer]) for kmer, c in counts.items()}
