"""Build the 4-layer hybrid multimodal knowledge graph."""
import logging
from typing import List, Dict, Optional
import networkx as nx

from src.kg.schema import (
    Entity, Assertion, SupportUnit, ProvenanceNode,
    Block, BlockType, Modality, SupportRole,
    NodeType, EdgeType,
)

logger = logging.getLogger(__name__)


class HybridKGBuilder:
    """Assembles the 4-layer VEGA-KG from extracted components.

    Layers:
    1. Global Entity & Ontology Layer
    2. Assertion Layer (text + visual)
    3. Multimodal Support Layer
    4. Structure / Provenance Layer
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.assertions: Dict[str, Assertion] = {}
        self.supports: Dict[str, SupportUnit] = {}
        self.provenance: Dict[str, ProvenanceNode] = {}

    def add_entity(self, entity: Entity):
        """Add entity to Layer 1."""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            node_type=NodeType.ENTITY.value,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            aliases=entity.aliases,
        )

    def add_assertion(self, assertion: Assertion):
        """Add assertion to Layer 2 and connect to entities."""
        self.assertions[assertion.id] = assertion
        self.graph.add_node(
            assertion.id,
            node_type=NodeType.ASSERTION.value,
            predicate=assertion.predicate,
            modality=assertion.modality.value,
            qualifiers=assertion.qualifiers,
            confidence=assertion.confidence,
            scope_atoms=assertion.scope_atoms,
        )

        # Connect to subject entity (Layer 1 -> Layer 2)
        if assertion.subject_id in self.entities:
            self.graph.add_edge(
                assertion.subject_id, assertion.id,
                edge_type=EdgeType.SUBJECT_OF.value,
            )

        # Connect to object entity (Layer 1 -> Layer 2)
        if not assertion.object_is_literal and assertion.object_id in self.entities:
            self.graph.add_edge(
                assertion.id, assertion.object_id,
                edge_type=EdgeType.OBJECT_OF.value,
            )

        # Connect to support units (Layer 2 -> Layer 3)
        for support_id in assertion.source_support_ids:
            if support_id in self.supports:
                self.graph.add_edge(
                    assertion.id, support_id,
                    edge_type=EdgeType.SUPPORTS.value,
                )

    def add_support(self, support: SupportUnit):
        """Add support unit to Layer 3."""
        self.supports[support.id] = support
        self.graph.add_node(
            support.id,
            node_type=NodeType.SUPPORT.value,
            unit_type=support.unit_type.value,
            role=support.role.value,
            content=support.content[:200],  # truncate for graph node
            image_path=support.image_path,
            page=support.page,
            doc_id=support.doc_id,
            section_path=support.section_path,
        )

    def add_provenance(self, provenance: ProvenanceNode):
        """Add provenance to Layer 4."""
        self.provenance[provenance.id] = provenance
        self.graph.add_node(
            provenance.id,
            node_type=NodeType.PROVENANCE.value,
            doc_id=provenance.doc_id,
            section_path=provenance.section_path,
            page=provenance.page,
        )

    def add_same_as_edge(self, entity_id_1: str, entity_id_2: str):
        """Add cross-document same_as edge between entities."""
        self.graph.add_edge(
            entity_id_1, entity_id_2,
            edge_type=EdgeType.SAME_AS.value,
        )
        self.graph.add_edge(
            entity_id_2, entity_id_1,
            edge_type=EdgeType.SAME_AS.value,
        )

    def add_scope_edge(self, support_id: str, scope_support_id: str):
        """Add has_scope edge from support to its scope carrier."""
        self.graph.add_edge(
            support_id, scope_support_id,
            edge_type=EdgeType.HAS_SCOPE.value,
        )

    def add_grounded_to_edge(self, support_id: str, provenance_id: str):
        """Add grounded_to edge from support to provenance."""
        self.graph.add_edge(
            support_id, provenance_id,
            edge_type=EdgeType.GROUNDED_TO.value,
        )

    def build_from_blocks(self, blocks: List[Block],
                          assertions: List[Assertion],
                          entities: Dict[str, Entity]):
        """Build the full graph from preprocessed data.

        Args:
            blocks: All blocks from preprocessing.
            assertions: All assertions (text + visual).
            entities: Entity inventory from linker.
        """
        logger.info(f"Building graph: {len(entities)} entities, "
                    f"{len(assertions)} assertions, {len(blocks)} blocks")

        # Pre-build block map once (avoid O(n²) in governing header lookup)
        block_map = {b.id: b for b in blocks}

        # Layer 1: Add entities
        for entity in entities.values():
            self.add_entity(entity)
        logger.info(f"Layer 1 done: {len(entities)} entities")

        # Layer 3: Create support units from blocks
        for i, block in enumerate(blocks):
            support = SupportUnit(
                id=block.id,
                unit_type=block.block_type,
                role=self._infer_role(block),
                content=block.text,
                image_path=block.image_path,
                bbox=block.bbox,
                page=block.page,
                doc_id=block.doc_id,
                section_path=block.section_path,
                governing_header=self._get_governing_header_text(block, block_map),
            )
            self.add_support(support)

            # Layer 4: Provenance
            prov = ProvenanceNode(
                id=f"prov_{block.id}",
                doc_id=block.doc_id,
                section_path=block.section_path,
                page=block.page,
                bbox=block.bbox,
            )
            self.add_provenance(prov)
            self.add_grounded_to_edge(block.id, prov.id)

            if (i + 1) % 500000 == 0:
                logger.info(f"Layer 3/4 progress: {i+1}/{len(blocks)} blocks")

        logger.info(f"Layer 3/4 done: {len(self.supports)} supports, {len(self.provenance)} provenance")

        # Layer 2: Add assertions
        for assertion in assertions:
            self.add_assertion(assertion)
        logger.info(f"Layer 2 done: {len(self.assertions)} assertions")

        # Add scope edges (support -> governing header support)
        for block in blocks:
            if block.parent_id and block.parent_id in block_map:
                parent = block_map[block.parent_id]
                if parent.block_type in (BlockType.TITLE, BlockType.HEADER):
                    if parent.id in self.supports:
                        self.add_scope_edge(block.id, parent.id)

        logger.info(
            f"Built KG: {len(self.entities)} entities, "
            f"{len(self.assertions)} assertions, "
            f"{len(self.supports)} supports, "
            f"{len(self.provenance)} provenance nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def _infer_role(self, block: Block) -> SupportRole:
        if block.block_type in (BlockType.TITLE, BlockType.HEADER):
            return SupportRole.SCOPE
        elif block.block_type == BlockType.CAPTION:
            return SupportRole.CONTEXTUAL
        elif block.block_type == BlockType.FOOTER:
            return SupportRole.EXCEPTION
        return SupportRole.PRIMARY

    def _get_governing_header_text(self, block: Block, block_map: Dict[str, Block]) -> str:
        if not block.parent_id:
            return ""
        current = block
        while current.parent_id:
            parent = block_map.get(current.parent_id)
            if parent is None:
                break
            if parent.block_type in (BlockType.TITLE, BlockType.HEADER):
                return parent.text.strip()
            current = parent
        return ""

    def get_entity_neighborhood(self, entity_id: str, hops: int = 2) -> nx.DiGraph:
        """Get the subgraph around an entity within N hops."""
        if entity_id not in self.graph:
            return nx.DiGraph()

        nodes = {entity_id}
        frontier = {entity_id}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            next_frontier -= nodes
            nodes.update(next_frontier)
            frontier = next_frontier

        return self.graph.subgraph(nodes).copy()

    def get_assertion_neighborhood(self, assertion_id: str) -> Dict:
        """Get the full context for an assertion: subject, object, supports, scope."""
        if assertion_id not in self.assertions:
            return {}

        assertion = self.assertions[assertion_id]
        result = {
            "assertion": assertion.to_dict(),
            "subject": None,
            "object": None,
            "supports": [],
            "scope_carriers": [],
        }

        # Subject entity
        if assertion.subject_id in self.entities:
            result["subject"] = self.entities[assertion.subject_id].to_dict()

        # Object entity
        if not assertion.object_is_literal and assertion.object_id in self.entities:
            result["object"] = self.entities[assertion.object_id].to_dict()

        # Support units
        for sid in assertion.source_support_ids:
            if sid in self.supports:
                support = self.supports[sid]
                result["supports"].append(support.to_dict())

                # Scope carriers
                for _, target, data in self.graph.edges(sid, data=True):
                    if data.get("edge_type") == EdgeType.HAS_SCOPE.value:
                        if target in self.supports:
                            result["scope_carriers"].append(
                                self.supports[target].to_dict()
                            )

        return result

    def save(self, path: str):
        """Save graph to file."""
        from src.utils.io_utils import save_pickle
        data = {
            "graph": self.graph,
            "entities": self.entities,
            "assertions": self.assertions,
            "supports": self.supports,
            "provenance": self.provenance,
        }
        save_pickle(data, path)
        logger.info(f"Saved KG to {path}")

    @classmethod
    def load(cls, path: str) -> "HybridKGBuilder":
        """Load graph from file."""
        from src.utils.io_utils import load_pickle
        data = load_pickle(path)
        builder = cls()
        builder.graph = data["graph"]
        builder.entities = data["entities"]
        builder.assertions = data["assertions"]
        builder.supports = data["supports"]
        builder.provenance = data["provenance"]
        return builder
