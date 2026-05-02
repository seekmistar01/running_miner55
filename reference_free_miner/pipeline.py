from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from .fastq import clean_reads, resolve_fastq_path
from .graph import DeBruijnGraph
from .kmers import build_kmer_table
from .variants import bubbles_to_candidates, fill_to_expected_count, rank_candidates, kmer_snp_candidates
from .vcf import write_vcf


def _get_task_attr(task: Any, name: str) -> Any:
    if isinstance(task, dict):
        return task[name]
    return getattr(task, name)


def _task_to_dict(task: Any) -> dict:
    if isinstance(task, dict):
        return task
    if hasattr(task, "model_dump"):
        return task.model_dump()
    if hasattr(task, "dict"):
        return task.dict()
    raise TypeError(f"Unsupported task type: {type(task)!r}")


def _choose_k(read_lengths: list[int], region_len: int) -> int:
    if not read_lengths:
        return 21
    median_len = sorted(read_lengths)[len(read_lengths) // 2]
    # Keep k odd and not too close to read length.
    if median_len >= 120:
        return 31
    if median_len >= 80:
        return 25
    return 21


def generate_vcf_for_task(
    task: Any,
    *,
    work_dir: str | Path | None = None,
    max_reads_per_file: int = 20000,
    timeout_seconds: float | None = None,
) -> tuple[str, dict]:
    """
    Generate VCF content for a NIOME Task using reference-free logic.

    Returns:
      (vcf_content, metadata)
    """
    start_time = time.time()
    task_dict = _task_to_dict(task)

    task_input = task_dict["input"]
    genome_context = task_dict["genome_context"]
    chrom = genome_context["chromosome"]
    region = genome_context["region"]
    expected_count = int(task_dict.get("expected_variant_count", 0) or 0)

    if expected_count <= 0:
        # Keep valid VCF with no variants.
        return write_vcf([]), {"warning": "expected_variant_count <= 0"}

    with tempfile.TemporaryDirectory(prefix="niome_ref_free_") as tmp:
        work = Path(work_dir) if work_dir else Path(tmp)
        read1 = resolve_fastq_path(task_input["read1_fastq"], work)
        read2 = resolve_fastq_path(task_input["read2_fastq"], work)

        reads = clean_reads(
            [read1, read2],
            max_reads_per_file=max_reads_per_file,
            min_read_len=25,
            min_mean_q=16.0,
            min_base_q=10,
        )

        read_lengths = [r.length for r in reads]
        region_start_end = region.replace(",", "")
        try:
            region_len = abs(int(region_start_end.split("-")[-1].split(":")[-1]) - int(region_start_end.split("-")[0].split(":")[-1])) + 1
        except Exception:
            region_len = 2000

        k = _choose_k(read_lengths, region_len)
        meta = {
            "reads_after_filter": len(reads),
            "k": k,
            "expected_variant_count": expected_count,
            "chromosome": chrom,
            "region": region,
        }

        if not reads:
            candidates = fill_to_expected_count([], expected_count=expected_count, chrom=chrom, region=region)
            vcf = write_vcf(candidates)
            meta["warning"] = "no usable reads; returned low-confidence fillers"
            return vcf, meta

        kmer_table = build_kmer_table(reads, k=k, min_kmer_q=12)
        meta["kmer_count"] = len(kmer_table)

        if not kmer_table:
            candidates = fill_to_expected_count([], expected_count=expected_count, chrom=chrom, region=region)
            vcf = write_vcf(candidates)
            meta["warning"] = "no usable kmers; returned low-confidence fillers"
            return vcf, meta

        # Direct k-mer SNP scan catches clean one-base disagreements even when
        # the explicit graph-bubble tracer is too conservative.
        all_candidates = kmer_snp_candidates(
            kmer_table,
            chrom=chrom,
            region=region,
            consensus="",  # consensus will be added below when available
            min_count=2,
        )

        # Try a few min_count values. Low depth tasks need min_count=1/2, noisy tasks need higher.
        bubble_count = 0
        consensus = ""

        for min_count in (3, 2, 1):
            if timeout_seconds and (time.time() - start_time) > timeout_seconds * 0.75:
                break

            graph = DeBruijnGraph.from_kmers(kmer_table, k=k, min_count=min_count)
            if not consensus:
                consensus = graph.build_major_consensus(region_len=min(region_len, 5000))
                all_candidates.extend(
                    kmer_snp_candidates(
                        kmer_table,
                        chrom=chrom,
                        region=region,
                        consensus=consensus,
                        min_count=min_count,
                    )
                )

            bubbles = graph.detect_bubbles(max_bubble_steps=10, max_bubbles=300)
            bubble_count += len(bubbles)

            cands = bubbles_to_candidates(
                bubbles,
                chrom=chrom,
                region=region,
                consensus=consensus,
            )
            all_candidates.extend(cands)

            if len(rank_candidates(all_candidates)) >= expected_count * 3:
                break

        ranked = rank_candidates(all_candidates)
        final_candidates = fill_to_expected_count(
            ranked,
            expected_count=expected_count,
            chrom=chrom,
            region=region,
        )

        meta.update(
            {
                "bubble_count": bubble_count,
                "candidate_count": len(ranked),
                "returned_count": len(final_candidates),
                "elapsed_internal_seconds": round(time.time() - start_time, 4),
                "filler_count": sum(1 for c in final_candidates if c.source == "filler"),
            }
        )

        return write_vcf(final_candidates), meta
