"""Qwen3-VL reader for generating answers from evidence packets."""
import logging
from typing import List, Dict, Optional

from src.reader.prompt_templates import format_qa_prompt
from src.reader.serializer import EvidenceSerializer

logger = logging.getLogger(__name__)


class QwenReader:
    """Reader that generates answers using Qwen3-VL from serialized evidence."""

    def __init__(self, vlm_client, token_budget: int = 16384,
                 max_images: int = 4, max_new_tokens: int = 256):
        """
        Args:
            vlm_client: LLM/VLM client with generate(prompt, image_path) method.
            token_budget: Max tokens for context.
            max_images: Max images to include.
            max_new_tokens: Max tokens to generate for answer.
        """
        self.vlm = vlm_client
        self.serializer = EvidenceSerializer(token_budget, max_images)
        self.max_new_tokens = max_new_tokens

    def answer(self, question: str, evidence_units: List[Dict],
               doc_meta: Dict = None) -> str:
        """Generate an answer from evidence.

        Args:
            question: The query.
            evidence_units: Retrieved evidence units (list of dicts).
            doc_meta: Optional document metadata.

        Returns:
            Generated answer string.
        """
        # Serialize evidence into context string
        context = self.serializer.serialize(evidence_units, doc_meta)

        # Format QA prompt
        prompt = format_qa_prompt(question=question, context=context)

        # Collect image paths for VLM
        image_paths = []
        for unit in evidence_units:
            if unit.get("image_path") and len(image_paths) < self.serializer.max_images:
                image_paths.append(unit["image_path"])

        # Generate answer
        # For now, pass first image if any (multi-image support depends on VLM)
        image_path = image_paths[0] if image_paths else None
        answer = self.vlm.generate(prompt, image_path=image_path)

        return answer.strip()
