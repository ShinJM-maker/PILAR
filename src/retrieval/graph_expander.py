"""Graph expansion from seed nodes."""
import logging
from collections import deque
from typing import List, Set
import networkx as nx

from src.kg.schema import EdgeType, NodeType

logger = logging.getLogger(__name__)


class GraphExpander:
    """Expand seed nodes via typed graph traversal.

    v2: SUPPORTS only, max_hops=2 (support → assertion → support).
    Bidirectional: follows both outgoing and incoming edges.
    """

    def __init__(self, graph: nx.DiGraph, max_hops: int = 2,
                 allowed_edges: List[str] = None):
        self.graph = graph
        self.max_hops = max_hops
        self.allowed_edges = set(allowed_edges or [EdgeType.SUPPORTS.value])

    def expand(self, seed_ids: List[str], max_nodes: int = 20) -> List[str]:
        """BFS expansion from seed nodes via allowed edges.

        Returns list of node IDs in the expanded neighborhood.
        """
        visited: Set[str] = set()
        result: List[str] = []
        queue = deque((sid, 0) for sid in seed_ids)

        while queue and len(result) < max_nodes:
            node_id, depth = queue.popleft()

            if node_id in visited:
                continue
            if node_id not in self.graph:
                continue

            visited.add(node_id)
            result.append(node_id)

            if depth >= self.max_hops:
                continue

            # Outgoing edges
            for _, neighbor, data in self.graph.edges(node_id, data=True):
                edge_type = data.get("edge_type", "")
                if edge_type in self.allowed_edges and neighbor not in visited:
                    queue.append((neighbor, depth + 1))

            # Incoming edges (reverse traversal for directed graph)
            for predecessor, _, data in self.graph.in_edges(node_id, data=True):
                edge_type = data.get("edge_type", "")
                if edge_type in self.allowed_edges and predecessor not in visited:
                    queue.append((predecessor, depth + 1))

        return result

    def get_support_units(self, node_ids: List[str]) -> List[str]:
        """Filter node IDs to only support-type nodes."""
        return [
            nid for nid in node_ids
            if self.graph.nodes.get(nid, {}).get("node_type") == NodeType.SUPPORT.value
        ]
