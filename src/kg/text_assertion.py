"""Text-based assertion extraction using LLM (Qwen3-8B)."""
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from src.kg.schema import Block, Assertion, BlockType, Modality
from src.kg.predicate_inventory import normalize_predicate, is_valid_predicate

logger = logging.getLogger(__name__)

# Load prompt template
_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "text_assertion_extract.txt"


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text()


class TextAssertionExtractor:
    """Extract structured assertions from text blocks using a text-only LLM."""

    def __init__(self, llm_client, max_assertions_per_block: int = 5):
        """
        Args:
            llm_client: An object with a `generate(prompt) -> str` method.
                       Can be a vLLM client, OpenAI-compatible client, etc.
            max_assertions_per_block: Max assertions to extract per block.
        """
        self.llm = llm_client
        self.max_assertions = max_assertions_per_block
        self.prompt_template = _load_prompt_template()
        self._assertion_counter = 0

    def extract(self, block: Block) -> List[Assertion]:
        """Extract assertions from a single text block."""
        if block.block_type not in (BlockType.PARAGRAPH, BlockType.CAPTION,
                                     BlockType.HEADER, BlockType.LIST):
            return []

        if not block.text.strip() or len(block.text.strip()) < 20:
            return []

        prompt = self.prompt_template.replace(
            "{section_path}", block.section_path or "N/A"
        ).replace(
            "{text}", block.text
        )

        try:
            response = self.llm.generate(prompt)
            raw_assertions = self._parse_response(response)
        except Exception as e:
            logger.warning(f"LLM extraction failed for {block.id}: {e}")
            return []

        # Convert to Assertion objects
        assertions = []
        for raw in raw_assertions[:self.max_assertions]:
            assertion = self._to_assertion(raw, block)
            if assertion:
                assertions.append(assertion)

        return assertions

    def extract_batch(self, blocks: List[Block]) -> Dict[str, List[Assertion]]:
        """Extract assertions from multiple blocks.

        Returns:
            Dict mapping block_id to list of assertions.
        """
        results = {}
        text_blocks = [b for b in blocks
                       if b.block_type in (BlockType.PARAGRAPH, BlockType.CAPTION,
                                            BlockType.HEADER, BlockType.LIST)
                       and b.text.strip() and len(b.text.strip()) >= 20]

        # Prepare all prompts
        prompts = []
        block_ids = []
        for block in text_blocks:
            prompt = self.prompt_template.replace(
                "{section_path}", block.section_path or "N/A"
            ).replace(
                "{text}", block.text
            )
            prompts.append(prompt)
            block_ids.append(block.id)

        # Batch generate
        if hasattr(self.llm, 'generate_batch'):
            responses = self.llm.generate_batch(prompts)
        else:
            responses = [self.llm.generate(p) for p in prompts]

        # Parse results
        block_map = {b.id: b for b in text_blocks}
        for block_id, response in zip(block_ids, responses):
            try:
                raw_assertions = self._parse_response(response)
                assertions = []
                for raw in raw_assertions[:self.max_assertions]:
                    assertion = self._to_assertion(raw, block_map[block_id])
                    if assertion:
                        assertions.append(assertion)
                results[block_id] = assertions
            except Exception as e:
                logger.warning(f"Failed to parse response for {block_id}: {e}")
                results[block_id] = []

        return results

    def _parse_response(self, response: str) -> List[dict]:
        """Parse LLM JSON response into list of raw assertion dicts."""
        response = response.strip()
        # Try to find JSON array in the response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.debug(f"JSON parse error: {json_str[:200]}")
            return []

    def _to_assertion(self, raw: dict, block: Block) -> Optional[Assertion]:
        """Convert raw parsed dict to Assertion object."""
        subject = raw.get("subject", "").strip()
        predicate = raw.get("predicate", "").strip()
        obj = raw.get("object", "").strip()

        if not subject or not predicate or not obj:
            return None

        # Normalize predicate
        predicate = normalize_predicate(predicate)

        # Filter grounded only
        if not raw.get("grounded", True):
            return None

        self._assertion_counter += 1
        assertion_id = f"ta_{self._assertion_counter:06d}"

        return Assertion(
            id=assertion_id,
            subject_id=subject,  # will be entity-linked later
            predicate=predicate,
            object_id=obj,  # will be entity-linked later
            object_is_literal=self._is_literal(obj),
            qualifiers=raw.get("qualifiers", {}),
            modality=Modality.TEXT,
            confidence=1.0,
            grounded=True,
            scope_atoms=block.section_path.split(" > ") if block.section_path else [],
            source_support_ids=[block.id],
        )

    @staticmethod
    def _is_literal(value: str) -> bool:
        """Check if a value is a literal (number, date, etc.) vs entity name."""
        # Simple heuristic: if it contains digits and is short, likely a literal
        if any(c.isdigit() for c in value) and len(value) < 50:
            return True
        return False
