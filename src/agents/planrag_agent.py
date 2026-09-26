"""PlanRAG agent: plan-then-retrieve."""
import logging
from typing import Dict, List
from src.agents.agent_interface import AgentInterface

logger = logging.getLogger(__name__)

PLAN_PROMPT = """/no_think
Generate a retrieval plan to answer the following question.
Break the question into sub-queries that can be searched independently.

Question: {question}

Output a JSON array of sub-queries:
["<sub-query 1>", "<sub-query 2>", ...]

Output ONLY the JSON array, no explanation."""


class PlanRAGAgent(AgentInterface):
    """PlanRAG: generate retrieval plan, then execute.

    1. Generate a plan (sequence of sub-queries)
    2. Execute each sub-query against the retriever
    3. Synthesize from all retrieved evidence
    """

    def __init__(self, retriever, reader, planning_llm=None,
                 max_subqueries: int = 5, top_k: int = 10):
        super().__init__(retriever, reader)
        self.planning_llm = planning_llm
        self.max_subqueries = max_subqueries
        self.top_k = top_k

    def answer(self, question: str) -> Dict:
        # Step 1: Generate retrieval plan
        sub_queries = self._generate_plan(question)

        # Step 2: Execute plan
        all_evidence = []
        existing_ids = set()
        for sq in sub_queries:
            evidence = self.retriever.retrieve(sq, top_k=self.top_k)
            for e in evidence:
                if e.get("id") not in existing_ids:
                    all_evidence.append(e)
                    existing_ids.add(e.get("id"))

        # Step 3: Synthesize answer
        answer_text = self.reader.answer(question, all_evidence)

        return {
            "answer": answer_text,
            "evidence": all_evidence,
            "metadata": {
                "agent": "planrag",
                "sub_queries": sub_queries,
                "num_evidence_units": len(all_evidence),
            },
        }

    def _generate_plan(self, question: str) -> List[str]:
        """Generate sub-queries for the retrieval plan."""
        if self.planning_llm:
            import json
            prompt = PLAN_PROMPT.replace("{question}", question)
            response = self.planning_llm.generate(prompt)
            try:
                start = response.find("[")
                end = response.rfind("]") + 1
                if start >= 0 and end > 0:
                    queries = json.loads(response[start:end])
                    return queries[:self.max_subqueries]
            except Exception:
                pass

        # Fallback: use original question + entity-focused variant
        return [question, f"entities mentioned in: {question}"]
