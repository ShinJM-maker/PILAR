"""ReAct-Lite agent: naive-first with selective repair, gate, verifier, and router."""
import logging
import re
import time
from typing import Dict, List, Set

from src.agents.agent_interface import AgentInterface

logger = logging.getLogger(__name__)

REACT_LITE_SYSTEM_PROMPT = """/no_think
You retrieve evidence to answer questions. Output exactly ONE line starting with SEARCH: or ANSWER:.

SEARCH: <query> - retrieve relevant evidence from the knowledge base.
ANSWER: <short answer> - give your final answer (a few words only).

Rules:
- You MUST start with SEARCH on the first step.
- After reviewing evidence, either SEARCH again with a different query or give ANSWER.
- Maximum {max_steps} steps. If unsure, output ANSWER: unanswerable.

Question: {question}

Evidence so far:
{evidence}

Output SEARCH: or ANSWER:"""

# ── Deny-list router patterns ──
TEMPORAL_PATTERNS = re.compile(
    r'\b(?:as\s+of|during|since|until)\b', re.IGNORECASE
)
LIST_PATTERNS = re.compile(
    r'\b(?:list|all\s+of|how\s+many|each)\b', re.IGNORECASE
)

# ── Condition-only verifier patterns ──
CONDITION_PATTERNS = re.compile(
    r'\b(?:as\s+of|during|since|until)\b', re.IGNORECASE
)


def _tokenize(s: str) -> Set[str]:
    return set(re.findall(r'\w+', s.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ReActLiteAgent(AgentInterface):
    """Naive-first ReAct-Lite with optional gate, verifier, and router.

    Flow:
    1. Naive pass: single retrieve + read (step 0)
    2. (Optional) Deny-list router: skip repair for temporal/list questions
    3. Repair loop: up to max_steps additional retrieval steps
    4. Final read with accumulated evidence
    5. (Optional) Supported-naive gate: keep naive answer if well-supported
    6. (Optional) Condition verifier: keep naive answer for temporal questions
    """

    def __init__(self, retriever, reader, reasoning_llm=None,
                 max_steps: int = 2, top_k: int = 10,
                 enable_naive_gate: bool = False,
                 enable_condition_verifier: bool = False,
                 enable_deny_router: bool = False,
                 detailed_logging: bool = False):
        super().__init__(retriever, reader)
        self.reasoning_llm = reasoning_llm
        self.max_steps = max_steps
        self.top_k = top_k
        self.enable_naive_gate = enable_naive_gate
        self.enable_condition_verifier = enable_condition_verifier
        self.enable_deny_router = enable_deny_router
        self.detailed_logging = detailed_logging

    def answer(self, question: str) -> Dict:
        step_logs = []
        all_evidence = []
        existing_ids: Set[str] = set()
        prev_queries: List[str] = []
        trace = []

        # ── Step 0: Naive pass (single retrieval) ──
        step_start = time.time()
        naive_evidence = self.retriever.retrieve(question, top_k=self.top_k)
        for e in naive_evidence:
            eid = e.get("id")
            if eid not in existing_ids:
                all_evidence.append(e)
                existing_ids.add(eid)
        prev_queries.append(question)

        naive_answer = self.reader.answer(question, all_evidence)
        naive_eu_ids = [e.get("id") for e in all_evidence]
        naive_page_ids = list(dict.fromkeys(
            eid.rsplit("_", 1)[0] if "_" in eid else eid
            for eid in naive_eu_ids
        ))
        naive_is_unanswerable = "unanswerable" in naive_answer.lower()

        step_log_0 = {
            "step": 0,
            "action": "naive_retrieve",
            "query": question,
            "new_eu_ids": naive_eu_ids,
            "packet_eu_count": len(all_evidence),
            "elapsed_s": round(time.time() - step_start, 3),
        }
        step_logs.append(step_log_0)
        trace.append({"step": 0, "decision": f"SEARCH: {question}", "naive_answer": naive_answer})

        # ── Deny-list router check ──
        repair_skipped = False
        router_reason = None
        if self.enable_deny_router:
            if TEMPORAL_PATTERNS.search(question):
                repair_skipped = True
                router_reason = "temporal_deny"
            elif LIST_PATTERNS.search(question):
                repair_skipped = True
                router_reason = "list_deny"

        # ── Repair loop (steps 1..max_steps) ──
        repair_ran = False
        if not repair_skipped:
            for step in range(1, self.max_steps + 1):
                step_start = time.time()

                evidence_text = self._format_evidence(all_evidence)
                prompt = REACT_LITE_SYSTEM_PROMPT.replace(
                    "{max_steps}", str(self.max_steps)
                ).replace(
                    "{question}", question
                ).replace(
                    "{evidence}", evidence_text or "None yet."
                )

                if self.reasoning_llm:
                    decision = self.reasoning_llm.generate(prompt)
                else:
                    decision = f"SEARCH: {question}"

                trace.append({"step": step, "decision": decision})

                step_log = {
                    "step": step,
                    "action": None,
                    "query": None,
                    "new_eu_ids": [],
                    "packet_eu_count": 0,
                    "novelty": {},
                    "elapsed_s": 0.0,
                }

                if "ANSWER:" in decision:
                    step_log["action"] = "answer"
                    step_log["packet_eu_count"] = len(all_evidence)
                    step_log["elapsed_s"] = round(time.time() - step_start, 3)
                    step_logs.append(step_log)
                    break

                # Extract search query
                if "SEARCH:" in decision:
                    query = decision.split("SEARCH:", 1)[1].strip()
                    if not query:
                        query = question
                    step_log["action"] = "search"
                else:
                    query = question
                    step_log["action"] = "fallback_search"

                step_log["query"] = query

                # Query novelty
                query_tokens = _tokenize(query)
                all_prev_tokens = set()
                for pq in prev_queries:
                    all_prev_tokens |= _tokenize(pq)
                step_log["novelty"] = {
                    "query_jaccard_vs_all_prev": round(_jaccard(query_tokens, all_prev_tokens), 4),
                    "new_token_ratio": round(
                        len(query_tokens - all_prev_tokens) / max(len(query_tokens), 1), 4
                    ),
                }
                prev_queries.append(query)

                # Retrieve
                new_evidence = self.retriever.retrieve(query, top_k=self.top_k)
                added_ids = []
                for e in new_evidence:
                    eid = e.get("id")
                    if eid not in existing_ids:
                        all_evidence.append(e)
                        existing_ids.add(eid)
                        added_ids.append(eid)

                step_log["new_eu_ids"] = added_ids
                step_log["packet_eu_count"] = len(all_evidence)
                step_log["novelty"]["new_eu_count"] = len(added_ids)
                step_log["elapsed_s"] = round(time.time() - step_start, 3)
                step_logs.append(step_log)
                repair_ran = True

        # ── Final read ──
        if repair_ran or repair_skipped:
            # If repair was skipped, final answer = naive answer
            # If repair ran, re-read with full evidence
            if repair_skipped:
                final_answer = naive_answer
            else:
                final_answer = self.reader.answer(question, all_evidence)
        else:
            final_answer = naive_answer

        # ── Supported-naive gate ──
        gate_fired = False
        gate_reason = None
        if self.enable_naive_gate and repair_ran and not repair_skipped:
            sufficient_support = (
                not naive_is_unanswerable
                and len(naive_page_ids) >= 2
            )
            if sufficient_support:
                # Keep naive answer unless repair added genuinely new evidence
                new_eu_after_naive = set()
                for sl in step_logs[1:]:  # skip step 0
                    for eid in sl.get("new_eu_ids", []):
                        new_eu_after_naive.add(eid)
                if len(new_eu_after_naive) == 0:
                    # No new evidence from repair → keep naive
                    final_answer = naive_answer
                    gate_fired = True
                    gate_reason = "no_new_evidence"
                # If new evidence was found, allow the repair answer

        # ── Condition-only verifier ──
        verifier_fired = False
        if self.enable_condition_verifier and repair_ran and not repair_skipped:
            if CONDITION_PATTERNS.search(question):
                final_answer = naive_answer
                verifier_fired = True

        # ── Build metadata ──
        metadata = {
            "agent": "react_lite",
            "num_steps": len(step_logs),
            "trace": trace,
            "num_evidence_units": len(all_evidence),
            "repair_skipped": repair_skipped,
            "router_reason": router_reason,
            "gate_fired": gate_fired,
            "gate_reason": gate_reason,
            "verifier_fired": verifier_fired,
            "naive_answer": naive_answer,
            "config": {
                "max_steps": self.max_steps,
                "enable_naive_gate": self.enable_naive_gate,
                "enable_condition_verifier": self.enable_condition_verifier,
                "enable_deny_router": self.enable_deny_router,
            },
        }

        if self.detailed_logging:
            metadata["step_logs"] = step_logs
            metadata["packet_eu_ids"] = [e.get("id") for e in all_evidence]
            metadata["naive_gate"] = {
                "naive_answer": naive_answer,
                "naive_answerability_flag": "unanswerable" if naive_is_unanswerable else "answerable",
                "naive_support_count": len(naive_page_ids),
                "naive_cited_eu_ids": naive_eu_ids,
                "naive_conflict_flag": 0,
            }

        return {
            "answer": final_answer,
            "evidence": all_evidence,
            "metadata": metadata,
        }

    def _format_evidence(self, evidence: List[Dict]) -> str:
        parts = []
        for e in evidence[:10]:
            content = (e.get("content", "") or e.get("text", ""))[:200]
            parts.append(f"[{e.get('id', '?')}] {content}")
        return "\n".join(parts)
