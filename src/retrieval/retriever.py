"""Unified retrieval interface for VEGA-KG."""
import logging
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import networkx as nx

from src.kg.schema import SupportUnit, NodeType, EdgeType
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.graph_expander import GraphExpander

logger = logging.getLogger(__name__)


class VEGAKGRetriever:
    """VEGA-KG v3a retrieval pipeline:
    1. Page-level BM25+Dense seed retrieval (wider pool: top_k * 3)
    2. Local continuity bonus (adjacent ±1 page, same section)
    3. Conservative KG expansion: alias grounded pages + same-assertion corroboration
    4. Expanded page direct relevance gate (dense re-score + threshold)
    5. Additive scoring with min-max normalization
    6. Conditional base-first packing (expanded only if competitive)
    """

    def __init__(self, graph: nx.DiGraph, supports: Dict[str, SupportUnit],
                 page_bm25: BM25Index, page_dense: DenseIndex,
                 page_chunks: Dict[str, Dict],
                 entities: Dict[str, dict],
                 assertions: Dict[str, dict],
                 config: dict):
        self.graph = graph
        self.supports = supports
        self.page_bm25 = page_bm25
        self.page_dense = page_dense
        self.page_chunks = page_chunks
        self.token_budget = config.get("token_budget", 4500)

        # v3a knobs
        self.initial_multiplier = config.get("initial_multiplier", 3)
        self.seed_top_n = config.get("seed_top_n", 3)
        self.seed_gate_ratio = config.get("seed_gate_ratio", 0.85)

        self.max_matched_entities = config.get("max_matched_entities", 2)
        self.alias_doc_df_threshold = config.get("alias_doc_df_threshold", 10)
        self.max_alias_pages_per_entity = config.get("max_alias_pages_per_entity", 2)

        self.graph_max_hops = config.get("expansion_hops", 2)
        self.graph_max_nodes = config.get("expansion_max_nodes", 20)
        self.allowed_expansion_edges = set(
            config.get("expansion_edges", [EdgeType.SUPPORTS.value])
        )
        # Both expansion paths discard any candidate outside the documents the base retriever already
        # surfaced, which makes cross-document corroboration unreachable however the graph is built.
        # Off by default so existing runs are bit-identical; set true to let the graph route between
        # documents, which is what a multi-hop benchmark needs.
        self.expansion_cross_doc = config.get("expansion_cross_doc", False)
        # BFS is fed only this many of a seed page's support units. A page carries ~99 supports, so the
        # default samples 10% of them; a chunk carries ~2 and the cap never binds.
        self.seed_supports_cap = config.get("seed_supports_cap", 10)

        self.alias_bonus = config.get("alias_bonus", 0.12)
        self.assertion_bonus = config.get("assertion_bonus", 0.08)
        self.max_total_bonus = config.get("max_total_bonus", 0.20)

        # Feature toggles for factorial experiments
        self.enable_local_bonus = config.get("enable_local_bonus", True)
        self.enable_expanded_gate = config.get("enable_expanded_gate", True)

        # Local continuity
        self.adjacent_page_bonus = config.get("adjacent_page_bonus", 0.04)
        self.same_section_bonus = config.get("same_section_bonus", 0.015)
        self.local_bonus_cap = config.get("local_bonus_cap", 0.05)

        # Expanded page filtering
        self.expanded_dense_threshold = config.get("expanded_dense_threshold", 0.35)
        self.expanded_accept_epsilon = config.get("expanded_accept_epsilon", 0.03)
        self.expanded_score_scale = config.get("expanded_score_scale", 0.45)
        self.expanded_score_mode = config.get("expanded_score_mode", "scaled")

        self.base_budget_ratio = config.get("base_budget_ratio", 0.70)
        self.max_expanded_pages = config.get("max_expanded_pages", 2)

        # Build support → page_chunk_id mapping
        self._support_to_chunk = {}
        self._chunk_to_supports = defaultdict(list)
        for sid, sup in supports.items():
            if sup.doc_id:
                chunk_id = f"{sup.doc_id}_p{sup.page:04d}"
                if chunk_id in page_chunks:
                    self._support_to_chunk[sid] = chunk_id
                    self._chunk_to_supports[chunk_id].append(sid)

        # Build doc_id → chunk_ids, chunk_id → doc_id mappings
        self._doc_to_chunks = defaultdict(list)
        self._chunk_to_doc_id = {}
        self._chunk_texts = {}
        for cid, chunk in page_chunks.items():
            doc_id = chunk.get("doc_id", "")
            if doc_id:
                self._doc_to_chunks[doc_id].append(cid)
                self._chunk_to_doc_id[cid] = doc_id
            self._chunk_texts[cid] = chunk.get("text", "")

        # Build section index for local continuity
        self._chunk_section_path = {}
        self._section_to_chunks = defaultdict(list)  # (doc_id, section) → [chunk_ids]
        for cid, chunk in page_chunks.items():
            sec = chunk.get("section_path", "")
            if sec:
                self._chunk_section_path[cid] = sec
                doc_id = chunk.get("doc_id", "")
                self._section_to_chunks[(doc_id, sec)].append(cid)

        # Build entity alias → entity_id mapping (strict matching)
        self._entity_aliases = defaultdict(list)
        self._alias_doc_ids = defaultdict(set)
        self._entity_doc_ids = defaultdict(set)

        for eid, entity in entities.items():
            aliases = entity.get("aliases", [entity.get("canonical_name", eid)])
            doc_sources = entity.get("doc_sources", [])
            for doc_id in doc_sources:
                self._entity_doc_ids[eid].add(doc_id)
            for alias in aliases:
                alias_norm = self._normalize_text(alias)
                if len(alias_norm) < 3:
                    continue
                self._entity_aliases[alias_norm].append(eid)
                self._alias_doc_ids[alias_norm].update(doc_sources)

        # Build entity → grounded support pages
        self._entity_grounded_pages = defaultdict(Counter)
        for aid, assertion in assertions.items():
            subj = assertion.get("subject_id", "")
            obj = assertion.get("object_id", "")
            for sid in assertion.get("source_support_ids", []):
                sup = supports.get(sid)
                if sup and sup.doc_id:
                    chunk_id = f"{sup.doc_id}_p{sup.page:04d}"
                    if chunk_id in page_chunks:
                        if subj:
                            self._entity_grounded_pages[subj][chunk_id] += 1
                            self._entity_doc_ids[subj].add(sup.doc_id)
                        if obj and not assertion.get("object_is_literal", False):
                            self._entity_grounded_pages[obj][chunk_id] += 1
                            self._entity_doc_ids[obj].add(sup.doc_id)

        # Graph expander: SUPPORTS only, max_hops=2
        self.expander = GraphExpander(
            graph, max_hops=self.graph_max_hops,
            allowed_edges=list(self.allowed_expansion_edges),
        )

        logger.info(f"VEGAKGRetriever v3a: {len(self._support_to_chunk)} supports mapped, "
                     f"{len(self._entity_aliases)} aliases, "
                     f"{len(self._entity_grounded_pages)} entities with grounded pages, "
                     f"initial_multiplier={self.initial_multiplier}")

    # ---- Helpers ----

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _minmax_normalize(scored: Dict[str, float]) -> Dict[str, float]:
        if not scored:
            return {}
        vals = list(scored.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-8:
            return {k: 1.0 for k in scored}
        return {k: (v - lo) / (hi - lo) for k, v in scored.items()}

    def _expanded_to_base_scale(self, dense_sim: float, base_lo: float, base_hi: float) -> float:
        """Score an expanded page on whatever scale base pages are ranked on.

        `scaled` is the shipped behaviour: a raw cosine similarity, discounted, competing against base
        scores that have been min-max normalized to [0,1]. The two are not the same quantity, so the
        comparison in `_pack_conditional` ("is this expanded page within epsilon of the base page it
        would displace?") is not meaningful.

        `comparable` reconstructs what the page's base score would have been -- it has no BM25 entry,
        so only the dense term contributes -- and applies the base pool's own min-max transform. An
        expanded page then wins a packing slot exactly when it would have outranked the displaced page
        had the base retriever surfaced it.
        """
        if self.expanded_score_mode == "comparable":
            raw = 0.6 * dense_sim
            if base_hi - base_lo < 1e-8:
                return 1.0 if raw >= base_hi else 0.0
            return max(0.0, min(1.0, (raw - base_lo) / (base_hi - base_lo)))
        return dense_sim * self.expanded_score_scale

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    def _adjacent_chunk_ids(self, chunk_id: str) -> List[str]:
        """Get ±1 page chunk IDs for a given chunk."""
        # chunk_id format: {doc_id}_p{page:04d}
        try:
            prefix, page_part = chunk_id.rsplit("_p", 1)
            page_num = int(page_part)
        except (ValueError, AttributeError):
            return []
        result = []
        for offset in [-1, 1]:
            adj_id = f"{prefix}_p{page_num + offset:04d}"
            if adj_id in self.page_chunks:
                result.append(adj_id)
        return result

    # ---- Entity Matching ----

    def _match_query_entities(self, query: str) -> List[str]:
        """Strict entity matching: no substring, whole-word/phrase only."""
        q_norm = self._normalize_text(query)
        q_pad = f" {q_norm} "
        q_tokens = set(q_norm.split())

        candidates = []

        for alias_norm, eids in self._entity_aliases.items():
            doc_df = len(self._alias_doc_ids.get(alias_norm, set()))
            if doc_df > self.alias_doc_df_threshold:
                continue

            alias_tokens = alias_norm.split()

            if len(alias_norm) <= 3 or len(alias_tokens) == 1:
                matched = alias_norm in q_tokens
            else:
                matched = f" {alias_norm} " in q_pad

            if not matched:
                continue

            sort_key = (len(alias_tokens), len(alias_norm), -doc_df)
            for eid in set(eids):
                candidates.append((eid, sort_key))

        candidates.sort(key=lambda x: x[1], reverse=True)

        out = []
        seen = set()
        for eid, _ in candidates:
            if eid in seen:
                continue
            out.append(eid)
            seen.add(eid)
            if len(out) >= self.max_matched_entities:
                break

        return out

    # ---- Local Continuity Bonus ----

    def _compute_local_bonus(
        self,
        seed_chunk_ids: set,
        base_candidate_ids: set,
        best_score: float,
        seed_scores: Dict[str, float],
    ) -> Counter:
        """Compute adjacency and same-section bonus for base candidates only."""
        local_bonus = Counter()

        for seed_id in seed_chunk_ids:
            seed_doc = self._chunk_to_doc_id.get(seed_id)
            if seed_doc is None:
                continue
            seed_weight = seed_scores.get(seed_id, 0.0) / max(best_score, 1e-8)

            # Adjacent pages (±1)
            for adj_id in self._adjacent_chunk_ids(seed_id):
                if adj_id in base_candidate_ids and adj_id not in seed_chunk_ids:
                    bonus = self.adjacent_page_bonus * seed_weight
                    local_bonus[adj_id] = max(local_bonus[adj_id], bonus)

            # Same section siblings
            sec = self._chunk_section_path.get(seed_id)
            if sec:
                for sib in self._section_to_chunks.get((seed_doc, sec), []):
                    if sib in base_candidate_ids and sib != seed_id:
                        bonus = self.same_section_bonus * seed_weight
                        local_bonus[sib] = max(local_bonus[sib], bonus)

        # Cap per page
        for cid in list(local_bonus):
            local_bonus[cid] = min(local_bonus[cid], self.local_bonus_cap)

        return local_bonus

    # ---- KG Expansion: Alias ----

    def _expand_alias_pages(
        self,
        matched_entity_ids: List[str],
        seed_doc_ids: set,
        seed_chunk_ids: set,
    ) -> Counter:
        """Alias expansion: only grounded support pages, same-doc, max per entity."""
        alias_votes = Counter()

        for eid in matched_entity_ids:
            kept = 0
            for chunk_id, _cnt in self._entity_grounded_pages.get(eid, Counter()).most_common():
                doc_id = self._chunk_to_doc_id.get(chunk_id)
                if doc_id is None:
                    continue
                if not self.expansion_cross_doc and doc_id not in seed_doc_ids:
                    continue
                if chunk_id in seed_chunk_ids:
                    continue

                alias_votes[chunk_id] += 1
                kept += 1
                if kept >= self.max_alias_pages_per_entity:
                    break

        return alias_votes

    # ---- KG Expansion: Graph ----

    def _expand_graph_pages(
        self,
        seed_chunk_ids: set,
        seed_doc_ids: set,
    ) -> Counter:
        """Graph expansion: same-assertion corroboration via SUPPORTS edges."""
        assertion_votes = Counter()

        seed_supports = []
        seen = set()
        for chunk_id in seed_chunk_ids:
            for sid in self._chunk_to_supports.get(chunk_id, []):
                if sid not in seen:
                    seed_supports.append(sid)
                    seen.add(sid)

        if not seed_supports:
            return assertion_votes

        expanded_nodes = self.expander.expand(
            seed_supports[:self.seed_supports_cap], max_nodes=self.graph_max_nodes
        )
        expanded_sids = self.expander.get_support_units(expanded_nodes)

        for sid in expanded_sids:
            chunk_id = self._support_to_chunk.get(sid)
            if chunk_id is None:
                continue
            doc_id = self._chunk_to_doc_id.get(chunk_id)
            if doc_id is None:
                continue
            if not self.expansion_cross_doc and doc_id not in seed_doc_ids:
                continue
            if chunk_id in seed_chunk_ids:
                continue

            assertion_votes[chunk_id] += 1

        return assertion_votes

    # ---- Expanded Page Relevance Gate ----

    def _gate_expanded_pages(
        self,
        query: str,
        expanded_chunk_ids: set,
        base_candidate_ids: set,
    ) -> Dict[str, float]:
        """Re-score expanded pages with direct dense similarity.
        Only pages NOT in base results are re-scored.
        Returns {chunk_id: direct_relevance_score} for pages passing threshold.
        """
        new_expanded = expanded_chunk_ids - base_candidate_ids
        if not new_expanded:
            return {}

        scored = self.page_dense.score_many(query, list(new_expanded))
        return {cid: s for cid, s in scored.items() if s >= self.expanded_dense_threshold}

    # ---- Packing ----

    def _pack_base_first(
        self,
        final_scored: Dict[str, float],
        base_candidate_ids: set,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """v2-style packing: base first, then unconditional expanded slots."""
        base_ranked = sorted(
            [(cid, final_scored[cid]) for cid in base_candidate_ids if cid in final_scored],
            key=lambda x: x[1], reverse=True,
        )
        expanded_ranked = sorted(
            [(cid, score) for cid, score in final_scored.items() if cid not in base_candidate_ids],
            key=lambda x: x[1], reverse=True,
        )

        selected = []
        selected_ids = set()
        used_tokens = 0
        base_budget = int(self.token_budget * self.base_budget_ratio)
        expanded_added = 0

        def try_add(chunk_id, score):
            nonlocal used_tokens
            if chunk_id in selected_ids:
                return False
            text = self._chunk_texts.get(chunk_id, "")
            if not text:
                return False
            est = self._estimate_tokens(text)
            if used_tokens + est > self.token_budget:
                return False
            selected.append((chunk_id, score))
            selected_ids.add(chunk_id)
            used_tokens += est
            return True

        for chunk_id, score in base_ranked[:top_k * 2]:
            if len(selected) >= top_k or used_tokens >= base_budget:
                break
            try_add(chunk_id, score)

        for chunk_id, score in expanded_ranked:
            if len(selected) >= top_k or expanded_added >= self.max_expanded_pages:
                break
            if try_add(chunk_id, score):
                expanded_added += 1

        for chunk_id, score in base_ranked:
            if len(selected) >= top_k or used_tokens >= self.token_budget:
                break
            try_add(chunk_id, score)

        return selected

    def _pack_conditional(
        self,
        final_scored: Dict[str, float],
        base_candidate_ids: set,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Base-first packing with conditional expanded slots.
        Expanded pages must be competitive with next-best base page.
        """
        base_ranked = sorted(
            [(cid, final_scored[cid]) for cid in base_candidate_ids if cid in final_scored],
            key=lambda x: x[1], reverse=True,
        )
        expanded_ranked = sorted(
            [(cid, score) for cid, score in final_scored.items() if cid not in base_candidate_ids],
            key=lambda x: x[1], reverse=True,
        )

        selected = []
        selected_ids = set()
        used_tokens = 0
        base_budget = int(self.token_budget * self.base_budget_ratio)
        expanded_added = 0

        def try_add(chunk_id, score):
            nonlocal used_tokens
            if chunk_id in selected_ids:
                return False
            text = self._chunk_texts.get(chunk_id, "")
            if not text:
                return False
            est = self._estimate_tokens(text)
            if used_tokens + est > self.token_budget:
                return False
            selected.append((chunk_id, score))
            selected_ids.add(chunk_id)
            used_tokens += est
            return True

        # Phase 1: base pages up to 70% budget
        for chunk_id, score in base_ranked[:top_k * 2]:
            if len(selected) >= top_k or used_tokens >= base_budget:
                break
            try_add(chunk_id, score)

        # Phase 2: expanded pages, only if competitive with next-best base
        next_best_base = None
        for cid, score in base_ranked:
            if cid not in selected_ids:
                next_best_base = score
                break

        for chunk_id, score in expanded_ranked:
            if len(selected) >= top_k or expanded_added >= self.max_expanded_pages:
                break
            # Skip if weaker than next base page minus epsilon
            if next_best_base is not None and score < next_best_base - self.expanded_accept_epsilon:
                continue
            if try_add(chunk_id, score):
                expanded_added += 1

        # Phase 3: fill remaining budget with more base pages
        for chunk_id, score in base_ranked:
            if len(selected) >= top_k or used_tokens >= self.token_budget:
                break
            try_add(chunk_id, score)

        return selected

    # ---- Main retrieve ----

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """v3a hybrid retrieval: wider pool + local continuity + KG + expanded gate."""

        # Step 1: Base retrieval (BM25 + Dense), wider pool
        top_n = top_k * self.initial_multiplier
        bm25_results = {did: s for did, s in self.page_bm25.search(query, top_n)}
        dense_results = {did: s for did, s in self.page_dense.search(query, top_n)}

        all_ids = set(bm25_results.keys()) | set(dense_results.keys())
        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0

        base_scored = {}
        for chunk_id in all_ids:
            bm25_norm = bm25_results.get(chunk_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            dense_score = dense_results.get(chunk_id, 0.0)
            base_scored[chunk_id] = 0.4 * bm25_norm + 0.6 * dense_score

        if not base_scored:
            return []

        # Keep the pre-normalization range: an expanded page has no BM25 entry, so putting it on the
        # same scale as base pages means running its dense score through this same transform rather
        # than letting a raw cosine compete against min-max normalized values.
        _base_vals = list(base_scored.values())
        base_lo, base_hi = min(_base_vals), max(_base_vals)

        # Min-max normalize
        base_scored = self._minmax_normalize(base_scored)
        base_candidate_ids = set(base_scored.keys())

        # Step 2: Seed selection (top N, gated)
        base_ranked = sorted(base_scored.items(), key=lambda x: x[1], reverse=True)
        best_score = base_ranked[0][1]

        seeds = []
        for cid, sc in base_ranked[:self.seed_top_n]:
            if sc >= self.seed_gate_ratio * best_score:
                seeds.append((cid, sc))

        seed_chunk_ids = {cid for cid, _ in seeds}
        seed_scores = {cid: sc for cid, sc in seeds}
        seed_doc_ids = {
            self._chunk_to_doc_id[cid]
            for cid in seed_chunk_ids
            if cid in self._chunk_to_doc_id
        }

        # Step 3: Local continuity bonus (adjacent ±1, same section)
        local_bonus = Counter()
        if self.enable_local_bonus:
            local_bonus = self._compute_local_bonus(
                seed_chunk_ids=seed_chunk_ids,
                base_candidate_ids=base_candidate_ids,
                best_score=best_score,
                seed_scores=seed_scores,
            )

        # Step 4: KG expansion (alias + graph)
        matched_entity_ids = self._match_query_entities(query)

        alias_votes = self._expand_alias_pages(
            matched_entity_ids=matched_entity_ids,
            seed_doc_ids=seed_doc_ids,
            seed_chunk_ids=seed_chunk_ids,
        )

        assertion_votes = self._expand_graph_pages(
            seed_chunk_ids=seed_chunk_ids,
            seed_doc_ids=seed_doc_ids,
        )

        # Step 5: Gate expanded pages with direct relevance (or accept all)
        kg_expanded_ids = (set(alias_votes.keys()) | set(assertion_votes.keys()))
        new_expanded_ids = kg_expanded_ids - base_candidate_ids
        expanded_direct_scores = {}
        gated_new_expanded = set()

        if self.enable_expanded_gate and new_expanded_ids:
            expanded_direct_scores = self._gate_expanded_pages(
                query=query,
                expanded_chunk_ids=kg_expanded_ids,
                base_candidate_ids=base_candidate_ids,
            )
            gated_new_expanded = set(expanded_direct_scores.keys())
        elif not self.enable_expanded_gate:
            # G=0: accept all new expanded pages (v2 behavior)
            gated_new_expanded = new_expanded_ids

        # Step 6: Final score = base + local + KG bonus
        # Base pages with KG bonus stay in base_candidate_ids (CRITICAL)
        all_candidate_ids = base_candidate_ids | gated_new_expanded

        final_scored = {}
        for chunk_id in all_candidate_ids:
            base = base_scored.get(chunk_id, 0.0)

            # For genuinely new expanded pages. Their raw dense similarity is discounted before it
            # competes with base pages, whose scores are min-max normalized to [0,1] -- so an expanded
            # page is handicapped roughly two-fold no matter how relevant it is. Configurable so the
            # handicap can be measured rather than assumed; the default is the shipped 0.45.
            if chunk_id in gated_new_expanded:
                if self.enable_expanded_gate and chunk_id in expanded_direct_scores:
                    base = self._expanded_to_base_scale(
                        expanded_direct_scores[chunk_id], base_lo, base_hi
                    )
                else:
                    # v2 behavior: flat bonus only, no dense re-score
                    base = 0.0

            bonus = 0.0

            # Local continuity
            if self.enable_local_bonus:
                bonus += local_bonus.get(chunk_id, 0.0)

            # KG bonuses
            kg_bonus = 0.0
            if alias_votes.get(chunk_id, 0) > 0:
                kg_bonus += self.alias_bonus
            if assertion_votes.get(chunk_id, 0) > 0:
                kg_bonus += self.assertion_bonus
            kg_bonus = min(kg_bonus, self.max_total_bonus)
            bonus += kg_bonus

            final_scored[chunk_id] = base + bonus

        # Step 7: Packing
        if self.enable_expanded_gate:
            # Conditional packing (expanded only if competitive)
            ranked = self._pack_conditional(
                final_scored=final_scored,
                base_candidate_ids=base_candidate_ids,
                top_k=top_k,
            )
        else:
            # v2-style packing (unconditional expanded slots)
            ranked = self._pack_base_first(
                final_scored=final_scored,
                base_candidate_ids=base_candidate_ids,
                top_k=top_k,
            )

        # Step 8: Determine adjacent chunk IDs (for role assignment)
        adjacent_chunk_ids = set()
        for seed_id in seed_chunk_ids:
            for adj_id in self._adjacent_chunk_ids(seed_id):
                if adj_id not in seed_chunk_ids:
                    adjacent_chunk_ids.add(adj_id)

        # Step 9: Build output with role metadata
        packet = []
        for chunk_id, score in ranked:
            chunk = self.page_chunks.get(chunk_id)
            if chunk:
                chunk_with_role = dict(chunk)
                # Assign role
                if chunk_id in seed_chunk_ids:
                    chunk_with_role["_role"] = "seed"
                elif chunk_id in adjacent_chunk_ids:
                    chunk_with_role["_role"] = "adjacent"
                elif chunk_id not in base_candidate_ids:
                    chunk_with_role["_role"] = "expanded"
                else:
                    chunk_with_role["_role"] = "other"
                chunk_with_role["_score"] = score
                packet.append(chunk_with_role)

        # Diagnostic logging
        n_base = sum(1 for cid, _ in ranked if cid in base_candidate_ids)
        n_exp = sum(1 for cid, _ in ranked if cid not in base_candidate_ids)
        n_local = sum(1 for cid in local_bonus if local_bonus[cid] > 0)
        n_exp_gated_out = len(new_expanded_ids) - len(gated_new_expanded)

        logger.debug(
            f"VEGA-KG v3a: base={len(base_candidate_ids)} | "
            f"seeds={len(seed_chunk_ids)} | "
            f"entities={len(matched_entity_ids)} | "
            f"alias_pages={len(alias_votes)} graph_pages={len(assertion_votes)} | "
            f"local_bonus={n_local} | "
            f"expanded_new={len(new_expanded_ids)} gated_pass={len(gated_new_expanded)} "
            f"gated_out={n_exp_gated_out} | "
            f"packed: {n_base}base + {n_exp}exp = {len(packet)} total"
        )

        return packet


class VEGAPageOnlyRetriever:
    """Ablation baseline: same retrieval + packing, NO KG expansion.
    Supports configurable initial_multiplier for pool size experiments.
    """

    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex,
                 chunks: Dict[str, Dict], token_budget: int = 4500,
                 initial_multiplier: int = 2):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.chunks = chunks
        self.token_budget = token_budget
        self.initial_multiplier = initial_multiplier

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """Page-level BM25+Dense retrieval (no expansion)."""
        top_n = top_k * self.initial_multiplier
        bm25_results = {did: s for did, s in self.bm25.search(query, top_n)}
        dense_results = {did: s for did, s in self.dense.search(query, top_n)}

        all_ids = set(bm25_results.keys()) | set(dense_results.keys())
        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0

        scored = {}
        for chunk_id in all_ids:
            bm25_norm = bm25_results.get(chunk_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            dense_score = dense_results.get(chunk_id, 0.0)
            scored[chunk_id] = 0.4 * bm25_norm + 0.6 * dense_score

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        packet = []
        tokens = 0
        for chunk_id, score in ranked:
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                chunk_tokens = len(chunk.get("text", "").split()) * 1.3
                if tokens + chunk_tokens > self.token_budget:
                    break
                packet.append(chunk)
                tokens += chunk_tokens

        return packet


class FlatChunkRetriever:
    """Baseline: simple flat chunk retrieval without KG."""

    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex,
                 chunks: Dict[str, Dict], token_budget: int = 16384):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.chunks = chunks
        self.token_budget = token_budget

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve top-k chunks."""
        bm25_results = {did: s for did, s in self.bm25.search(query, top_k * 2)}
        dense_results = {did: s for did, s in self.dense.search(query, top_k * 2)}

        all_ids = set(bm25_results.keys()) | set(dense_results.keys())
        max_bm25 = max(bm25_results.values()) if bm25_results else 1.0

        scored = []
        for doc_id in all_ids:
            bm25_norm = bm25_results.get(doc_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
            dense_score = dense_results.get(doc_id, 0.0)
            score = 0.4 * bm25_norm + 0.6 * dense_score
            scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        packet = []
        tokens = 0
        for doc_id, score in scored[:top_k]:
            if doc_id in self.chunks:
                chunk = self.chunks[doc_id]
                chunk_tokens = len(chunk.get("text", "").split()) * 1.3
                if tokens + chunk_tokens > self.token_budget:
                    break
                packet.append(chunk)
                tokens += chunk_tokens

        return packet
