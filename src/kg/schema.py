"""VEGA-KG 4-layer hybrid graph schema definitions."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class BlockType(str, Enum):
    TITLE = "Title"
    HEADER = "Header"
    PARAGRAPH = "Paragraph"
    TABLE = "Table"
    FIGURE = "Figure"
    CAPTION = "Caption"
    LIST = "List"
    FOOTER = "Footer"


class Modality(str, Enum):
    TEXT = "text"
    VISUAL = "visual"


class SupportRole(str, Enum):
    PRIMARY = "primary"
    CONTEXTUAL = "contextual"
    SCOPE = "scope"
    EXCEPTION = "exception"


class NodeType(str, Enum):
    ENTITY = "entity"
    ASSERTION = "assertion"
    SUPPORT = "support"
    PROVENANCE = "provenance"


class EdgeType(str, Enum):
    SUBJECT_OF = "subject_of"
    OBJECT_OF = "object_of"
    SUPPORTS = "supports"
    HAS_SCOPE = "has_scope"
    HAS_EXCEPTION = "has_exception"
    GROUNDED_TO = "grounded_to"
    SAME_AS = "same_as"
    PARENT_OF = "parent_of"  # in DHP tree


# --- Layer 1: Global Entity & Ontology ---

@dataclass
class Entity:
    id: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    entity_type: str = "Thing"  # Person, Organization, Location, Concept, ...
    doc_sources: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "canonical_name": self.canonical_name,
            "aliases": self.aliases, "entity_type": self.entity_type,
            "doc_sources": self.doc_sources, "description": self.description,
        }


# --- Layer 2: Assertion ---

@dataclass
class Assertion:
    id: str
    subject_id: str
    predicate: str
    object_id: str  # entity ID or literal value
    object_is_literal: bool = False
    qualifiers: Dict[str, str] = field(default_factory=dict)
    modality: Modality = Modality.TEXT
    confidence: float = 1.0
    grounded: bool = True
    scope_atoms: List[str] = field(default_factory=list)  # section path atoms
    source_support_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "subject_id": self.subject_id,
            "predicate": self.predicate, "object_id": self.object_id,
            "object_is_literal": self.object_is_literal,
            "qualifiers": self.qualifiers, "modality": self.modality.value,
            "confidence": self.confidence, "grounded": self.grounded,
            "scope_atoms": self.scope_atoms,
            "source_support_ids": self.source_support_ids,
        }


# --- Layer 3: Multimodal Support ---

@dataclass
class SupportUnit:
    id: str
    unit_type: BlockType  # Text, Table, Figure, etc.
    role: SupportRole = SupportRole.PRIMARY
    content: str = ""  # OCR text or linearized table
    image_path: Optional[str] = None  # path to cropped image
    bbox: Optional[Tuple[float, float, float, float]] = None
    page: int = 0
    doc_id: str = ""
    section_path: str = ""
    governing_header: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "unit_type": self.unit_type.value,
            "role": self.role.value, "content": self.content,
            "image_path": self.image_path,
            "bbox": list(self.bbox) if self.bbox else None,
            "page": self.page, "doc_id": self.doc_id,
            "section_path": self.section_path,
            "governing_header": self.governing_header,
        }


# --- Layer 4: Provenance ---

@dataclass
class ProvenanceNode:
    id: str
    doc_id: str
    section_path: str
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    cell_ids: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "doc_id": self.doc_id,
            "section_path": self.section_path, "page": self.page,
            "bbox": list(self.bbox) if self.bbox else None,
            "cell_ids": self.cell_ids,
        }


# --- Document Block (preprocessing output) ---

@dataclass
class Block:
    id: str
    doc_id: str
    page: int
    block_type: BlockType
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text: str = ""
    confidence: float = 0.0
    image_path: Optional[str] = None  # for Table/Figure crops
    parent_id: Optional[str] = None  # DHP parent
    children_ids: List[str] = field(default_factory=list)
    section_path: str = ""
    depth: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "doc_id": self.doc_id, "page": self.page,
            "block_type": self.block_type.value,
            "bbox": list(self.bbox), "text": self.text,
            "confidence": self.confidence, "image_path": self.image_path,
            "parent_id": self.parent_id, "children_ids": self.children_ids,
            "section_path": self.section_path, "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        d = d.copy()
        d["block_type"] = BlockType(d["block_type"])
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)
