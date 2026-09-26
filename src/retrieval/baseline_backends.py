"""Baseline KG backend retrievers: MS GraphRAG, LightRAG, SimpleDoc, MultiDocFusion."""
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from pathlib import Path
import numpy as np
import networkx as nx

from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex

logger = logging.getLogger(__name__)


class MSGraphRAGRetriever:
    """MS GraphRAG-style baseline.

    Builds community graph from entities, generates community summaries,
    retrieves relevant communities + their member chunks.
    """

    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex,
                 chunks: Dict[str, Dict], entities: Dict[str, Dict],
                 token_budget: int = 16384):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.chunks = chunks
        self.token_budget = token_budget

        # Build community graph
        self.graph = nx.Graph()
        self.communities = {}
        self.community_summaries = {}
        self._build_communities(entities)

    def _build_communities(self, entities: Dict[str, Dict]):
        """Build entity co-occurrence graph and detect communities."""
        import ahocorasick
        chunk_entities = defaultdict(set)
        entity_chunks = defaultdict(set)

        # Build alias -> entity_id mapping
        alias_to_eids = defaultdict(list)
        for eid, entity in entities.items():
            for alias in entity.get("aliases", [entity.get("canonical_name", eid)]):
                alias_lower = alias.lower().strip()
                if len(alias_lower) >= 2:
                    alias_to_eids[alias_lower].append(eid)

        # Build Aho-Corasick automaton for fast multi-pattern matching
        logger.info(f"GraphRAG: building automaton with {len(alias_to_eids)} patterns for {len(self.chunks)} chunks")

        if alias_to_eids:
            automaton = ahocorasick.Automaton()
            for alias, eids in alias_to_eids.items():
                automaton.add_word(alias, (alias, eids))
            automaton.make_automaton()

            # Search all chunks efficiently
            for idx, (cid, chunk) in enumerate(self.chunks.items()):
                text = chunk.get("text", "").lower()
                for _, (alias, eids) in automaton.iter(text):
                    for eid in eids:
                        chunk_entities[cid].add(eid)
                        entity_chunks[eid].add(cid)
                if (idx + 1) % 10000 == 0:
                    logger.info(f"GraphRAG entity matching: {idx+1}/{len(self.chunks)} chunks")

        # Filter out very common entities (noise) - keep only entities in <= 500 chunks
        entity_freq = {eid: len(cids) for eid, cids in entity_chunks.items()}
        frequent_entities = {eid for eid, freq in entity_freq.items() if freq > 500}
        if frequent_entities:
            logger.info(f"GraphRAG: filtering {len(frequent_entities)} high-frequency entities")

        # Add edges between co-occurring entities (limit per chunk to avoid O(n²))
        for cid, ents in chunk_entities.items():
            ents_list = [e for e in ents if e not in frequent_entities]
            if len(ents_list) > 20:  # Limit to top 20 per chunk
                ents_list = ents_list[:20]
            for i in range(len(ents_list)):
                for j in range(i + 1, len(ents_list)):
                    if self.graph.has_edge(ents_list[i], ents_list[j]):
                        self.graph[ents_list[i]][ents_list[j]]["weight"] += 1
                    else:
                        self.graph.add_edge(ents_list[i], ents_list[j], weight=1)

        logger.info(f"GraphRAG: co-occurrence graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        # Detect communities using Louvain (with resolution parameter to speed up)
        if len(self.graph.nodes) > 0:
            try:
                communities = nx.community.louvain_communities(self.graph, seed=42, resolution=2.0)
                for i, comm in enumerate(communities):
                    for node in comm:
                        self.communities[node] = i
                # Build community summaries (just concatenate entity names)
                comm_entities = defaultdict(list)
                for eid, cid in self.communities.items():
                    name = entities.get(eid, {}).get("canonical_name", eid)
                    comm_entities[cid].append(name)
                for cid, names in comm_entities.items():
                    self.community_summaries[cid] = ", ".join(names[:20])
            except Exception as e:
                logger.warning(f"Community detection failed: {e}")

        # Build community index for retrieval
        self._community_chunks = defaultdict(set)
        self._chunk_communities = defaultdict(set)  # Reverse mapping
        for eid, comm_id in self.communities.items():
            for cid in entity_chunks.get(eid, []):
                self._community_chunks[comm_id].add(cid)
                self._chunk_communities[cid].add(comm_id)

        logger.info(f"GraphRAG: {len(self.graph.nodes)} entities, "
                    f"{len(set(self.communities.values()))} communities")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve via community matching + chunk retrieval."""
        # First: standard BM25+Dense retrieval
        bm25_results = {did: s for did, s in self.bm25.search(query, top_k * 2)}
        dense_results = {did: s for did, s in self.dense.search(query, top_k * 2)}

        # Score chunks
        all_ids = set(bm25_results.keys()) | set(dense_results.keys())
        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0

        scored = {}
        for doc_id in all_ids:
            bm25_norm = bm25_results.get(doc_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            dense_score = dense_results.get(doc_id, 0.0)
            scored[doc_id] = 0.4 * bm25_norm + 0.6 * dense_score

        # Community expansion: add chunks from same communities as top seeds
        top_seeds = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:5]
        for seed_id, seed_score in top_seeds:
            # Find communities this seed chunk belongs to via precomputed mapping
            seed_comms = self._chunk_communities.get(seed_id, set())
            for comm_id in seed_comms:
                for cid in self._community_chunks.get(comm_id, []):
                    if cid not in scored:
                        scored[cid] = seed_score * 0.5  # community bonus

        # Sort and assemble
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        packet = []
        tokens = 0
        for doc_id, score in ranked[:top_k * 2]:
            if doc_id in self.chunks:
                chunk = self.chunks[doc_id]
                chunk_tokens = len(chunk.get("text", "").split()) * 1.3
                if tokens + chunk_tokens > self.token_budget:
                    break
                packet.append(chunk)
                tokens += chunk_tokens

        return packet


class LightRAGRetriever:
    """LightRAG-style baseline.

    Simple entity-relation triples, entity-centric retrieval.
    """

    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex,
                 chunks: Dict[str, Dict], assertions: List[Dict],
                 token_budget: int = 16384):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.chunks = chunks
        self.token_budget = token_budget

        # Build entity-to-chunk mapping from assertions
        self._entity_chunks = defaultdict(set)
        self._entity_relations = defaultdict(list)
        for a in assertions:
            subj = a.get("subject_id", "")
            obj = a.get("object_id", "")
            pred = a.get("predicate", "")
            sources = a.get("source_support_ids", [])
            for src in sources:
                self._entity_chunks[subj].add(src)
                self._entity_chunks[obj].add(src)
            self._entity_relations[subj].append((pred, obj, sources))
            self._entity_relations[obj].append((pred, subj, sources))

        logger.info(f"LightRAG: {len(self._entity_chunks)} entities tracked")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Entity-centric retrieval: find entities in query, then expand."""
        # Standard retrieval
        bm25_results = {did: s for did, s in self.bm25.search(query, top_k)}
        dense_results = {did: s for did, s in self.dense.search(query, top_k)}

        all_ids = set(bm25_results.keys()) | set(dense_results.keys())
        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0

        scored = {}
        for doc_id in all_ids:
            bm25_norm = bm25_results.get(doc_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            dense_score = dense_results.get(doc_id, 0.0)
            scored[doc_id] = 0.4 * bm25_norm + 0.6 * dense_score

        # Entity expansion: find entities mentioned in query
        query_lower = query.lower()
        for entity, chunk_ids in self._entity_chunks.items():
            if entity.lower() in query_lower:
                # Add all chunks related to this entity
                for cid in chunk_ids:
                    if cid not in scored:
                        scored[cid] = 0.3
                # Also add 1-hop relations
                for pred, related, sources in self._entity_relations.get(entity, []):
                    for src in sources:
                        if src not in scored:
                            scored[src] = 0.2

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        packet = []
        tokens = 0
        for doc_id, score in ranked[:top_k * 2]:
            if doc_id in self.chunks:
                chunk = self.chunks[doc_id]
                chunk_tokens = len(chunk.get("text", "").split()) * 1.3
                if tokens + chunk_tokens > self.token_budget:
                    break
                packet.append(chunk)
                tokens += chunk_tokens

        return packet


class ColPaliPrecomputedScores:
    """Lookup table for pre-computed ColPali query-page scores.

    Loads scores from JSON: {sample_id: [[chunk_id, score], ...]}
    Generated by scripts/precompute_colpali.py.
    """

    def __init__(self, scores_path: str = None):
        self._scores: Dict[str, List[Tuple[str, float]]] = {}
        self._query_to_sid: Dict[str, str] = {}

        if scores_path:
            self.load(scores_path)

    def load(self, scores_path: str):
        import json
        with open(scores_path, "r") as f:
            raw = json.load(f)
        for sid, pairs in raw.items():
            self._scores[sid] = [(cid, score) for cid, score in pairs]
        logger.info(f"ColPaliScores: loaded pre-computed scores for {len(self._scores)} queries")

    def set_query_mapping(self, samples: list):
        """Map sample questions to sample IDs for lookup."""
        for s in samples:
            self._query_to_sid[s["question"]] = s["id"]

    def search(self, query: str, top_k: int = 20, sample_id: str = None) -> List[Tuple[str, float]]:
        """Look up pre-computed scores for this query."""
        sid = sample_id or self._query_to_sid.get(query)
        if sid and sid in self._scores:
            return self._scores[sid][:top_k]
        return []


class SimpleDocRetriever:
    """SimpleDoc baseline.

    Hybrid retrieval combining:
      1. ColPali visual embeddings (page image → multi-vector MaxSim)
      2. VLM text descriptions (page image → description → dense search)
      3. Original text BM25+Dense

    Requires pre-generated data from scripts/generate_page_descriptions.py.
    """

    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex,
                 chunks: Dict[str, Dict], token_budget: int = 16384,
                 colpali_index: 'ColPaliPrecomputedScores' = None,
                 desc_index: DenseIndex = None,
                 descriptions: Dict[str, str] = None,
                 alpha_text: float = 0.4, alpha_visual: float = 0.35,
                 alpha_desc: float = 0.25):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.chunks = chunks
        self.token_budget = token_budget
        self.colpali_index = colpali_index
        self.desc_index = desc_index
        self.descriptions = descriptions or {}
        self.alpha_text = alpha_text
        self.alpha_visual = alpha_visual
        self.alpha_desc = alpha_desc

        n_desc = len(self.descriptions)
        logger.info(f"SimpleDoc: {len(chunks)} chunks, {n_desc} descriptions, "
                    f"colpali={'loaded' if colpali_index else 'none'}, "
                    f"desc_index={'loaded' if desc_index else 'none'}, "
                    f"alpha=(text={alpha_text}, visual={alpha_visual}, desc={alpha_desc})")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """3-channel hybrid retrieval."""
        # Channel 1: Text BM25 + Dense on original page text
        bm25_results = {did: s for did, s in self.bm25.search(query, top_k * 2)}
        dense_results = {did: s for did, s in self.dense.search(query, top_k * 2)}

        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0
        text_scored = {}
        for doc_id in set(bm25_results.keys()) | set(dense_results.keys()):
            bm25_norm = bm25_results.get(doc_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            dense_score = dense_results.get(doc_id, 0.0)
            text_scored[doc_id] = 0.4 * bm25_norm + 0.6 * dense_score

        # Channel 2: ColPali visual embeddings (MaxSim)
        visual_scored = {}
        if self.colpali_index is not None:
            visual_results = self.colpali_index.search(query, top_k * 2)
            for doc_id, score in visual_results:
                visual_scored[doc_id] = score

        # Channel 3: Description Dense
        desc_scored = {}
        if self.desc_index is not None:
            desc_results = self.desc_index.search(query, top_k * 2)
            for doc_id, score in desc_results:
                desc_scored[doc_id] = score

        # Merge: normalize each channel to [0, 1], then weighted sum
        all_ids = set(text_scored.keys()) | set(visual_scored.keys()) | set(desc_scored.keys())

        max_text = max(text_scored.values()) if text_scored else 1.0
        max_visual = max(visual_scored.values()) if visual_scored else 1.0
        max_desc = max(desc_scored.values()) if desc_scored else 1.0

        # Adjust weights when channels are missing
        active_weight = 0.0
        if text_scored:
            active_weight += self.alpha_text
        if visual_scored:
            active_weight += self.alpha_visual
        if desc_scored:
            active_weight += self.alpha_desc
        if active_weight == 0:
            active_weight = 1.0

        scored = {}
        for doc_id in all_ids:
            t = (text_scored.get(doc_id, 0.0) / max_text * self.alpha_text) if text_scored else 0.0
            v = (visual_scored.get(doc_id, 0.0) / max_visual * self.alpha_visual) if visual_scored else 0.0
            d = (desc_scored.get(doc_id, 0.0) / max_desc * self.alpha_desc) if desc_scored else 0.0
            scored[doc_id] = (t + v + d) / active_weight

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        packet = []
        tokens = 0
        for doc_id, score in ranked[:top_k * 2]:
            if doc_id in self.chunks:
                chunk = self.chunks[doc_id]
                chunk_tokens = len(chunk.get("text", "").split()) * 1.3
                if tokens + chunk_tokens > self.token_budget:
                    break
                packet.append(chunk)
                tokens += chunk_tokens

        return packet


class MultiDocFusionRetriever:
    """MultiDocFusion baseline (formerly SimpleDoc).

    Document-level retrieval with section awareness.
    Groups chunks by document, retrieves whole document sections.
    """

    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex,
                 chunks: Dict[str, Dict], token_budget: int = 16384):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.chunks = chunks
        self.token_budget = token_budget

        # Group chunks by document and section
        self._doc_chunks = defaultdict(list)
        self._section_chunks = defaultdict(list)
        for cid, chunk in chunks.items():
            doc_id = chunk.get("doc_id", "")
            section = chunk.get("section_path", "")
            self._doc_chunks[doc_id].append(cid)
            if section:
                self._section_chunks[f"{doc_id}::{section}"].append(cid)

        logger.info(f"MultiDocFusion: {len(self._doc_chunks)} docs, "
                    f"{len(self._section_chunks)} sections")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Section-aware retrieval."""
        bm25_results = {did: s for did, s in self.bm25.search(query, top_k)}
        dense_results = {did: s for did, s in self.dense.search(query, top_k)}

        all_ids = set(bm25_results.keys()) | set(dense_results.keys())
        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0

        scored = {}
        for doc_id in all_ids:
            bm25_norm = bm25_results.get(doc_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            dense_score = dense_results.get(doc_id, 0.0)
            scored[doc_id] = 0.4 * bm25_norm + 0.6 * dense_score

        # Section expansion: add sibling chunks from same section
        top_seeds = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:10]
        for seed_id, seed_score in top_seeds:
            chunk = self.chunks.get(seed_id, {})
            doc_id = chunk.get("doc_id", "")
            section = chunk.get("section_path", "")
            if section:
                key = f"{doc_id}::{section}"
                for sibling_id in self._section_chunks.get(key, []):
                    if sibling_id not in scored:
                        scored[sibling_id] = seed_score * 0.7  # section sibling bonus

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        packet = []
        tokens = 0
        for doc_id, score in ranked[:top_k * 2]:
            if doc_id in self.chunks:
                chunk = self.chunks[doc_id]
                chunk_tokens = len(chunk.get("text", "").split()) * 1.3
                if tokens + chunk_tokens > self.token_budget:
                    break
                packet.append(chunk)
                tokens += chunk_tokens

        return packet
