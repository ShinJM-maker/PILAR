"""AutoGen multi-agent conversation framework."""
import logging
from typing import Dict, List
from src.agents.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class AutoGenAgent(AgentInterface):
    """Multi-agent conversation: retriever agent + reasoner agent.

    The retriever agent queries the KG based on reasoner's requests.
    The reasoner agent synthesizes evidence and produces the answer.
    """

    def __init__(self, retriever, reader, reasoning_llm=None,
                 max_turns: int = 10, top_k: int = 10):
        super().__init__(retriever, reader)
        self.reasoning_llm = reasoning_llm
        self.max_turns = max_turns
        self.top_k = top_k

    def answer(self, question: str) -> Dict:
        """Simulate multi-agent conversation."""
        all_evidence = []
        existing_ids = set()
        conversation = []

        # Turn 1: Reasoner analyzes the question
        conversation.append({
            "role": "reasoner",
            "content": f"I need to answer: {question}. Let me search for relevant evidence."
        })

        # Turn 2: Retriever does initial search
        evidence = self.retriever.retrieve(question, top_k=self.top_k)
        for e in evidence:
            if e.get("id") not in existing_ids:
                all_evidence.append(e)
                existing_ids.add(e.get("id"))

        conversation.append({
            "role": "retriever",
            "content": f"Found {len(evidence)} relevant evidence units."
        })

        # Additional turns with reasoning LLM if available
        if self.reasoning_llm:
            for turn in range(2, self.max_turns):
                evidence_summary = self._summarize_evidence(all_evidence)
                prompt = (
                    f"/no_think\nQuestion: {question}\n\n"
                    f"Evidence so far:\n{evidence_summary}\n\n"
                    f"Do you need more evidence? If yes, output SEARCH: <query>. "
                    f"If you have enough, output DONE."
                )
                response = self.reasoning_llm.generate(prompt)

                if "DONE" in response:
                    break
                elif "SEARCH:" in response:
                    query = response.split("SEARCH:", 1)[1].strip()
                    new_evidence = self.retriever.retrieve(query, top_k=self.top_k)
                    for e in new_evidence:
                        if e.get("id") not in existing_ids:
                            all_evidence.append(e)
                            existing_ids.add(e.get("id"))
                    conversation.append({"role": "retriever", "content": f"Found {len(new_evidence)} more units."})

        # Final answer
        answer_text = self.reader.answer(question, all_evidence)

        return {
            "answer": answer_text,
            "evidence": all_evidence,
            "metadata": {
                "agent": "autogen",
                "num_turns": len(conversation),
                "conversation": conversation,
                "num_evidence_units": len(all_evidence),
            },
        }

    def _summarize_evidence(self, evidence: List[Dict]) -> str:
        parts = []
        for e in evidence[:10]:
            content = (e.get("content", "") or e.get("text", ""))[:150]
            parts.append(f"- [{e.get('id', '?')}] {content}")
        return "\n".join(parts)
