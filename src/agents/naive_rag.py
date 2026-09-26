"""Naive RAG: single-step retrieve-then-read."""
from typing import Dict
from src.agents.agent_interface import AgentInterface


class NaiveRAGAgent(AgentInterface):
    """Single-step retrieve-then-read agent.

    1. Retrieve top-K evidence units from the KG backend.
    2. Pack into context.
    3. Reader generates answer.
    """

    def __init__(self, retriever, reader, top_k: int = 20):
        super().__init__(retriever, reader)
        self.top_k = top_k

    def answer(self, question: str) -> Dict:
        # Step 1: Retrieve
        evidence = self.retriever.retrieve(question, top_k=self.top_k)

        # Step 2: Read
        answer_text = self.reader.answer(question, evidence)

        return {
            "answer": answer_text,
            "evidence": evidence,
            "metadata": {
                "agent": "naive_rag",
                "num_evidence_units": len(evidence),
            },
        }
