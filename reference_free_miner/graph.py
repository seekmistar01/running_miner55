from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, Optional

from .kmers import KmerStats


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    base: str
    kmer: str
    count: int
    mean_q: float


@dataclass(slots=True)
class PathTrace:
    end_node: str
    seq: str
    support: int
    mean_q: float
    edge_count: int


@dataclass(slots=True)
class Bubble:
    start_node: str
    end_node: str
    seq1: str
    seq2: str
    support1: int
    support2: int
    q1: float
    q2: float


@dataclass
class DeBruijnGraph:
    k: int
    outgoing: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))
    incoming: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def from_kmers(
        cls,
        kmers: dict[str, KmerStats],
        *,
        k: int,
        min_count: int = 2,
    ) -> "DeBruijnGraph":
        graph = cls(k=k)
        for kmer, stat in kmers.items():
            if stat.count < min_count:
                continue
            src = kmer[:-1]
            dst = kmer[1:]
            edge = Edge(
                src=src,
                dst=dst,
                base=kmer[-1],
                kmer=kmer,
                count=stat.count,
                mean_q=stat.mean_q,
            )
            graph.outgoing[src].append(edge)
            graph.incoming[dst].append(edge)

        for node in list(graph.outgoing):
            graph.outgoing[node].sort(key=lambda e: (e.count, e.mean_q), reverse=True)
        return graph

    def build_major_consensus(self, region_len: int = 2000) -> str:
        """
        Greedy highest-coverage path for a rough coordinate scaffold.

        This is not a true assembly; it is only used for estimating bubble offsets
        inside genome_context.region.
        """
        if not self.outgoing:
            return ""

        all_nodes = set(self.outgoing.keys()) | set(self.incoming.keys())
        source_nodes = [n for n in all_nodes if len(self.incoming.get(n, [])) == 0 and self.outgoing.get(n)]
        if not source_nodes:
            source_nodes = [n for n in self.outgoing if self.outgoing[n]]

        def node_score(n: str) -> int:
            return sum(e.count for e in self.outgoing.get(n, []))

        start = max(source_nodes, key=node_score)
        seq = start
        node = start
        seen = set()
        while len(seq) < region_len and node in self.outgoing and node not in seen:
            seen.add(node)
            edge = self.outgoing[node][0]
            seq += edge.base
            node = edge.dst
        return seq

    def _trace_paths_from_edge(
        self,
        edge: Edge,
        *,
        max_steps: int = 12,
        max_paths: int = 32,
    ) -> dict[str, PathTrace]:
        """
        Trace paths starting with one outgoing edge.

        Returns best path trace by endpoint node.
        """
        traces: dict[str, PathTrace] = {}
        stack: list[tuple[str, str, int, float, int, set[str]]] = [
            (edge.dst, edge.src + edge.base, edge.count, edge.mean_q, 1, {edge.src})
        ]

        while stack and len(traces) < max_paths:
            node, seq, support, q_sum, edge_count, seen = stack.pop()

            # stop at a join after at least one extension
            if edge_count > 1 and len(self.incoming.get(node, [])) > 1:
                mean_q = q_sum / max(edge_count, 1)
                prev = traces.get(node)
                if prev is None or support > prev.support:
                    traces[node] = PathTrace(node, seq, support, mean_q, edge_count)
                continue

            if edge_count >= max_steps:
                mean_q = q_sum / max(edge_count, 1)
                prev = traces.get(node)
                if prev is None or support > prev.support:
                    traces[node] = PathTrace(node, seq, support, mean_q, edge_count)
                continue

            outs = self.outgoing.get(node, [])
            if not outs:
                mean_q = q_sum / max(edge_count, 1)
                traces[node] = PathTrace(node, seq, support, mean_q, edge_count)
                continue

            # Avoid explosion: keep the strongest few edges.
            for nxt in outs[:4]:
                if nxt.dst in seen:
                    continue
                stack.append(
                    (
                        nxt.dst,
                        seq + nxt.base,
                        min(support, nxt.count),
                        q_sum + nxt.mean_q,
                        edge_count + 1,
                        seen | {node},
                    )
                )

        return traces

    def detect_bubbles(
        self,
        *,
        max_bubble_steps: int = 12,
        max_bubbles: int = 200,
    ) -> list[Bubble]:
        """Find simple split/rejoin bubbles."""
        bubbles: list[Bubble] = []
        for start, outs in self.outgoing.items():
            if len(outs) < 2:
                continue

            # Strongest branches first.
            outs = outs[:5]
            path_maps = [(edge, self._trace_paths_from_edge(edge, max_steps=max_bubble_steps)) for edge in outs]

            for (edge_a, paths_a), (edge_b, paths_b) in combinations(path_maps, 2):
                common_ends = set(paths_a) & set(paths_b)
                if not common_ends:
                    continue

                # Pick the common end with the highest combined support.
                end = max(common_ends, key=lambda n: paths_a[n].support + paths_b[n].support)
                pa = paths_a[end]
                pb = paths_b[end]
                if pa.seq == pb.seq:
                    continue

                bubbles.append(
                    Bubble(
                        start_node=start,
                        end_node=end,
                        seq1=pa.seq,
                        seq2=pb.seq,
                        support1=pa.support,
                        support2=pb.support,
                        q1=pa.mean_q,
                        q2=pb.mean_q,
                    )
                )
                if len(bubbles) >= max_bubbles:
                    return bubbles
        return bubbles
