"""ReAct agent: iterative reasoning-action loop."""
import hashlib
import logging
import re
import time
from typing import Dict, List, Set
from src.agents.agent_interface import AgentInterface

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """/no_think
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


def _tokenize(s: str) -> Set[str]:
    """Simple word tokenization for novelty metrics."""
    return set(re.findall(r'\w+', s.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def get_evidence_at_step_internal(step_logs: List[Dict], all_evidence: List[Dict],
                                   max_step: int) -> List[Dict]:
    """Reconstruct evidence packet at a given step from step_logs.

    Internal helper used during logging (not for replay script).
    """
    included_ids = set()
    for sl in step_logs:
        if sl["step"] > max_step:
            break
        for eid in sl.get("new_eu_ids", []):
            included_ids.add(eid)
    return [e for e in all_evidence if e.get("id") in included_ids]


class ReActAgent(AgentInterface):
    """ReAct: interleave reasoning and action steps.

    The agent decides whether to:
    - THINK: reason about what information is needed
    - SEARCH: query the retriever for more evidence
    - ANSWER: produce a final answer
    """

    def __init__(self, retriever, reader, reasoning_llm=None,
                 max_steps: int = 5, top_k: int = 10,
                 detailed_logging: bool = False,
                 prompt_horizon: int | None = None,
                 log_naive_gate: bool = True):
        super().__init__(retriever, reader)
        self.reasoning_llm = reasoning_llm
        self.max_steps = max_steps
        # A live stopping audit must not also change the policy prompt.  When
        # supplied, prompt_horizon stays fixed while max_steps is the external
        # execution stop.  The legacy behavior remains the default.
        self.prompt_horizon = max_steps if prompt_horizon is None else prompt_horizon
        self.top_k = top_k
        self.detailed_logging = detailed_logging
        self.log_naive_gate = log_naive_gate

    def answer(self, question: str) -> Dict:
        all_evidence = []
        trace = []
        step_logs = []  # detailed per-step logs for replay
        existing_ids: Set[str] = set()
        prev_queries: List[str] = []

        for step in range(self.max_steps):
            step_start = time.time()

            # Format prompt with accumulated evidence
            evidence_text = self._format_evidence(all_evidence)
            prompt = REACT_SYSTEM_PROMPT.replace(
                "{max_steps}", str(self.prompt_horizon)
            ).replace(
                "{question}", question
            ).replace(
                "{evidence}", evidence_text or "None yet."
            )

            # Get agent decision
            if self.reasoning_llm:
                decision = self.reasoning_llm.generate(prompt)
            else:
                if step == 0:
                    decision = f"SEARCH: {question}"
                elif step < self.max_steps - 1 and len(all_evidence) < self.top_k:
                    decision = f"SEARCH: {question}"
                else:
                    decision = "ANSWER: based on evidence"

            trace.append({"step": step, "decision": decision})

            # --- Step log entry ---
            step_log = {
                "step": step,
                "policy_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "decision": decision,
                "action": None,        # "search" | "answer" | "fallback_search"
                "query": None,
                "retrieved_eu_ids": [],
                "retrieved_eu_scores": [],
                "retrieved_page_ids": [],
                "new_eu_ids": [],       # actually added (after dedup)
                "new_page_ids": [],
                "packet_eu_ids": [],    # cumulative after this step
                "packet_eu_count": 0,
                "novelty": {},
                "elapsed_s": 0.0,
            }

            if "ANSWER:" in decision:
                step_log["action"] = "answer"
                step_log["packet_eu_ids"] = [e.get("id") for e in all_evidence]
                step_log["packet_eu_count"] = len(all_evidence)
                step_log["elapsed_s"] = time.time() - step_start
                step_logs.append(step_log)
                break

            # Determine query
            if "SEARCH:" in decision:
                query = decision.split("SEARCH:", 1)[1].strip()
                if not query:
                    query = question
                step_log["action"] = "search"
            else:
                query = question
                step_log["action"] = "fallback_search"

            step_log["query"] = query

            # Compute query novelty
            query_tokens = _tokenize(query)
            if prev_queries:
                all_prev_tokens = set()
                for pq in prev_queries:
                    all_prev_tokens |= _tokenize(pq)
                step_log["novelty"] = {
                    "query_jaccard_vs_all_prev": round(_jaccard(query_tokens, all_prev_tokens), 4),
                    "query_jaccard_vs_prev": round(_jaccard(query_tokens, _tokenize(prev_queries[-1])), 4),
                    "new_token_ratio": round(
                        len(query_tokens - all_prev_tokens) / max(len(query_tokens), 1), 4
                    ),
                }
            else:
                step_log["novelty"] = {
                    "query_jaccard_vs_all_prev": 0.0,
                    "query_jaccard_vs_prev": 0.0,
                    "new_token_ratio": 1.0,
                }
            prev_queries.append(query)

            # Retrieve
            new_evidence = self.retriever.retrieve(query, top_k=self.top_k)

            # Log retrieved results (before dedup)
            step_log["retrieved_eu_ids"] = [e.get("id") for e in new_evidence]
            step_log["retrieved_eu_scores"] = [
                round(e.get("_score", 0.0), 4) for e in new_evidence
            ]
            step_log["retrieved_page_ids"] = list(dict.fromkeys(
                e.get("id", "").rsplit("_", 1)[0] if "_" in e.get("id", "") else e.get("id", "")
                for e in new_evidence
            ))

            # Deduplicate and add
            added_ids = []
            added_page_ids = set()
            for e in new_evidence:
                eid = e.get("id")
                if eid not in existing_ids:
                    all_evidence.append(e)
                    existing_ids.add(eid)
                    added_ids.append(eid)
                    page_id = eid.rsplit("_", 1)[0] if "_" in eid else eid
                    added_page_ids.add(page_id)

            step_log["new_eu_ids"] = added_ids
            step_log["new_page_ids"] = sorted(added_page_ids)
            step_log["packet_eu_ids"] = [e.get("id") for e in all_evidence]
            step_log["packet_eu_count"] = len(all_evidence)

            # Evidence novelty (vs previous packet)
            prev_packet_size = len(all_evidence) - len(added_ids)
            step_log["novelty"]["new_eu_count"] = len(added_ids)
            step_log["novelty"]["new_eu_ratio"] = round(
                len(added_ids) / max(len(new_evidence), 1), 4
            )
            # Page-set Jaccard
            prev_page_set = set()
            for e in all_evidence[:prev_packet_size]:
                eid = e.get("id", "")
                prev_page_set.add(eid.rsplit("_", 1)[0] if "_" in eid else eid)
            retrieved_page_set = set(step_log["retrieved_page_ids"])
            step_log["novelty"]["page_jaccard_vs_prev_packet"] = round(
                _jaccard(retrieved_page_set, prev_page_set), 4
            ) if prev_page_set else 0.0

            step_log["elapsed_s"] = round(time.time() - step_start, 3)
            step_logs.append(step_log)

        # Final answer using reader
        answer_text = self.reader.answer(question, all_evidence)

        # Build metadata
        metadata = {
            "agent": "react",
            "num_steps": len(trace),
            "max_steps_budget": self.max_steps,
            "prompt_horizon": self.prompt_horizon,
            "trace": trace,
            "num_evidence_units": len(all_evidence),
        }

        if self.detailed_logging:
            metadata["step_logs"] = step_logs
            metadata["packet_eu_ids"] = [e.get("id") for e in all_evidence]

            if self.log_naive_gate:
                # Optional gate diagnostic. It performs an extra reader call and
                # is disabled when runtime is itself an audited outcome.
                step0_evidence = get_evidence_at_step_internal(step_logs, all_evidence, 0)
                naive_answer = self.reader.answer(question, step0_evidence)
                naive_eu_ids = [e.get("id") for e in step0_evidence]
                naive_page_ids = list(dict.fromkeys(
                    eid.rsplit("_", 1)[0] if "_" in eid else eid
                    for eid in naive_eu_ids
                ))
                naive_is_unanswerable = "unanswerable" in naive_answer.lower()

                metadata["naive_gate"] = {
                    "naive_answer": naive_answer,
                    "naive_answerability_flag": "unanswerable" if naive_is_unanswerable else "answerable",
                    "naive_support_count": len(naive_page_ids),
                    "naive_cited_eu_ids": naive_eu_ids,
                    "naive_packet_eu_ids": naive_eu_ids,
                    "naive_conflict_flag": 0,  # E4-v0: always 0
                }

        return {
            "answer": answer_text,
            "evidence": all_evidence,
            "metadata": metadata,
        }

    def _format_evidence(self, evidence: List[Dict]) -> str:
        parts = []
        for e in evidence[:10]:  # limit to avoid prompt overflow
            content = (e.get("content", "") or e.get("text", ""))[:200]
            parts.append(f"[{e.get('id', '?')}] {content}")
        return "\n".join(parts)
