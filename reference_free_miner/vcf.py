from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from .variants import VariantCandidate


def write_vcf(candidates: Iterable[VariantCandidate], *, source: str = "niome_reference_free_miner") -> str:
    """Return VCF text."""
    lines: list[str] = [
        "##fileformat=VCFv4.2",
        f"##source={source}",
        f"##fileDate={datetime.now(timezone.utc).strftime('%Y%m%d')}",
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read/k-mer depth supporting candidate">',
        '##INFO=<ID=AD,Number=1,Type=Integer,Description="Approximate alternate allele support">',
        '##INFO=<ID=AF,Number=1,Type=Float,Description="Approximate alternate allele frequency">',
        '##INFO=<ID=SRC,Number=1,Type=String,Description="Candidate source: bubble, kmer_snp, or filler">',
        '##INFO=<ID=LO,Number=1,Type=Integer,Description="Local offset estimated from reference-free consensus">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]

    for c in sorted(candidates, key=lambda x: (x.chrom, x.pos, x.ref, x.alt)):
        info = (
            f"DP={int(c.depth)};"
            f"AD={int(c.alt_depth)};"
            f"AF={c.allele_frequency:.4f};"
            f"SRC={c.source};"
            f"LO={int(c.local_offset)}"
        )
        filt = "PASS"
        lines.append(
            f"{c.chrom}\t{int(c.pos)}\t.\t{c.ref}\t{c.alt}\t{c.qual:.2f}\t{filt}\t{info}"
        )

    return "\n".join(lines) + "\n"
