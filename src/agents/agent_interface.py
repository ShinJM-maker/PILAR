"""Abstract agent interface for VEGA-KG experiments."""
from abc import ABC, abstractmethod
from typing import List, Dict


class AgentInterface(ABC):
    """Base class for all agent frameworks."""

    def __init__(self, retriever, reader):
        """
        Args:
            retriever: Object with retrieve(query, top_k) -> List[Dict] method.
            reader: Object with answer(question, evidence_units) -> str method.
        """
        self.retriever = retriever
        self.reader = reader

    @abstractmethod
    def answer(self, question: str) -> Dict:
        """Generate an answer for the given question.

        Returns:
            Dict with keys: "answer", "evidence", "metadata"
        """
        pass
