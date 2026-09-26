"""Entity linking: extract mentions, normalize, and link to canonical entities."""
import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

from src.kg.schema import Entity, Assertion, Block, BlockType

logger = logging.getLogger(__name__)


class EntityLinker:
    """5-step entity linking pipeline.

    1. Harvest candidate mentions from blocks
    2. Normalize lexical variants
    3. Dense matching against canonical inventory (with cached embeddings)
    4. Type constraint filtering
    5. Assign canonical IDs
    """

    def __init__(self, link_threshold: float = 0.75,
                 cross_doc_threshold: float = 0.8):
        self.link_threshold = link_threshold
        self.cross_doc_threshold = cross_doc_threshold
        self.entity_inventory: Dict[str, Entity] = {}
        self._entity_counter = 0
        self._embedder = None
        self._alias_to_entity: Dict[str, str] = {}
        # Cached embeddings for entity inventory
        self._entity_ids_list: List[str] = []
        self._entity_embs: Optional[np.ndarray] = None
        self._pending_embs: List[np.ndarray] = []  # Buffer before consolidation
        # Normalized name -> entity_id for fast exact matching
        self._norm_name_to_entity: Dict[str, str] = {}

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._embedder

    def link_assertions(self, assertions: List[Assertion],
                        blocks: Dict[str, Block]) -> List[Assertion]:
        """Link subject_id and object_id in assertions to canonical entities.

        Modifies assertions in-place and returns them.
        Also updates the entity inventory.
        """
        # Collect all unique mention strings
        mentions = set()
        for a in assertions:
            mentions.add(a.subject_id)
            if not a.object_is_literal:
                mentions.add(a.object_id)

        logger.info(f"Entity linking: {len(mentions)} unique mentions from {len(assertions)} assertions")

        # Link each mention (batch-encode all mentions first for efficiency)
        mention_to_entity = {}
        mention_list = list(mentions)

        # Pre-encode all mentions at once
        embedder = self._get_embedder()
        normalized_mentions = [self._normalize(m) for m in mention_list]
        mention_embs = embedder.encode(normalized_mentions, normalize_embeddings=True,
                                        batch_size=256, show_progress_bar=True)

        for i, mention in enumerate(mention_list):
            normalized = normalized_mentions[i]

            # Check alias cache first
            if normalized in self._alias_to_entity:
                mention_to_entity[mention] = self._alias_to_entity[normalized]
                continue

            # Try exact match
            exact = self._norm_name_to_entity.get(normalized)
            if exact:
                entity = self.entity_inventory[exact]
                if mention not in entity.aliases:
                    entity.aliases.append(mention)
                self._alias_to_entity[normalized] = exact
                mention_to_entity[mention] = exact
                continue

            # Dense matching against cached entity embeddings
            best_match = self._find_best_match_cached(mention_embs[i:i+1])

            if best_match:
                entity = self.entity_inventory[best_match]
                if mention not in entity.aliases:
                    entity.aliases.append(mention)
                self._alias_to_entity[normalized] = entity.id
                self._norm_name_to_entity[normalized] = entity.id
                mention_to_entity[mention] = entity.id
            else:
                entity = self._create_entity(mention, normalized, mention_embs[i])
                mention_to_entity[mention] = entity.id

            if (i + 1) % 50000 == 0:
                logger.info(f"Entity linking progress: {i+1}/{len(mention_list)} mentions, "
                           f"{len(self.entity_inventory)} entities so far")

        logger.info(f"Entity linking done: {len(self.entity_inventory)} entities from {len(mention_list)} mentions")

        # Update assertions with canonical entity IDs
        for a in assertions:
            if a.subject_id in mention_to_entity:
                a.subject_id = mention_to_entity[a.subject_id]
            if not a.object_is_literal and a.object_id in mention_to_entity:
                a.object_id = mention_to_entity[a.object_id]

        return assertions

    def _normalize(self, text: str) -> str:
        """Step 2: Normalize lexical variants."""
        text = text.strip().lower()
        # Remove articles
        text = re.sub(r'^(the|a|an)\s+', '', text)
        # Remove parenthetical info
        text = re.sub(r'\s*\(.*?\)\s*', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _consolidate_embs(self):
        """Consolidate pending embeddings into the main array."""
        if not self._pending_embs:
            return
        new_embs = np.array(self._pending_embs)
        if self._entity_embs is None:
            self._entity_embs = new_embs
        else:
            self._entity_embs = np.vstack([self._entity_embs, new_embs])
        self._pending_embs = []

    def _find_best_match_cached(self, query_emb: np.ndarray) -> Optional[str]:
        """Step 3: Dense matching using cached entity embeddings."""
        # Consolidate pending embeddings periodically
        if len(self._pending_embs) >= 1000:
            self._consolidate_embs()

        if self._entity_embs is None and not self._pending_embs:
            return None

        best_score = -1.0
        best_id = None

        # Search in consolidated embeddings
        if self._entity_embs is not None:
            similarities = (query_emb @ self._entity_embs.T)[0]
            idx = int(similarities.argmax())
            if similarities[idx] > best_score:
                best_score = float(similarities[idx])
                best_id = self._entity_ids_list[idx]

        # Search in pending embeddings
        if self._pending_embs:
            pending_arr = np.array(self._pending_embs)
            offset = len(self._entity_ids_list) - len(self._pending_embs)
            sims = (query_emb @ pending_arr.T)[0]
            idx = int(sims.argmax())
            if sims[idx] > best_score:
                best_score = float(sims[idx])
                best_id = self._entity_ids_list[offset + idx]

        if best_score >= self.link_threshold:
            return best_id

        return None

    def _create_entity(self, mention: str, normalized: str,
                       mention_emb: np.ndarray) -> Entity:
        """Create a new canonical entity and update embedding cache."""
        self._entity_counter += 1
        entity_id = f"e_{self._entity_counter:06d}"

        entity = Entity(
            id=entity_id,
            canonical_name=mention,
            aliases=[mention],
            entity_type=self._infer_type(mention),
        )
        self.entity_inventory[entity_id] = entity
        self._alias_to_entity[normalized] = entity_id
        self._norm_name_to_entity[normalized] = entity_id

        # Add to pending buffer (consolidated periodically)
        self._entity_ids_list.append(entity_id)
        self._pending_embs.append(mention_emb.reshape(-1))

        return entity

    def _infer_type(self, mention: str) -> str:
        """Simple heuristic type inference."""
        # Numbers/dates
        if re.match(r'^[\d.,]+$', mention):
            return "Value"
        if re.match(r'^\d{4}$', mention):
            return "Year"
        # Capitalized multi-word -> likely proper noun
        words = mention.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            return "Entity"
        return "Thing"

    def add_cross_doc_links(self) -> List[Tuple[str, str]]:
        """Add same_as edges between entities across documents.

        Returns list of (entity_id_1, entity_id_2) pairs.
        Uses chunked numpy operations to avoid O(n²) Python loops.
        """
        self._consolidate_embs()
        links = []
        entities = list(self.entity_inventory.values())
        if len(entities) < 2 or self._entity_embs is None:
            return links

        embs = self._entity_embs
        n = len(entities)
        chunk_size = 2000

        # Pre-build entity type array for vectorized type checking
        entity_types = [e.entity_type for e in entities]

        logger.info(f"Cross-doc linking: {n} entities, threshold={self.cross_doc_threshold}")

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_embs = embs[start:end]
            # Compare chunk against all entities after start
            rest_embs = embs[start:]
            sims = chunk_embs @ rest_embs.T  # (chunk_size, n-start)

            # Use numpy to find pairs above threshold (avoid Python nested loop)
            for i_local in range(end - start):
                # Only check j > i (upper triangle)
                row_sims = sims[i_local, i_local + 1:]
                above_threshold = np.where(row_sims >= self.cross_doc_threshold)[0]

                i_global = start + i_local
                for j_offset in above_threshold:
                    j_global = start + i_local + 1 + j_offset
                    if entity_types[i_global] == entity_types[j_global]:
                        links.append((entities[i_global].id, entities[j_global].id))

            if start > 0 and start % 10000 == 0:
                logger.info(f"Cross-doc linking progress: {start}/{n}, {len(links)} links so far")

        logger.info(f"Cross-doc linking done: {len(links)} links")
        return links
