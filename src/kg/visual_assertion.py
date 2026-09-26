"""VLM-driven visual assertion extraction from document image crops."""
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from src.kg.schema import Block, Assertion, BlockType, Modality
from src.kg.predicate_inventory import normalize_predicate

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "visual_assertion_extract.txt"


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text()


class VisualAssertionExtractor:
    """Extract structured assertions from visual elements using VLM (Qwen3-VL)."""

    def __init__(self, vlm_client, max_assertions_per_block: int = 5):
        """
        Args:
            vlm_client: An object with a `generate(prompt, image_path=None) -> str` method.
            max_assertions_per_block: Max assertions per visual block.
        """
        self.vlm = vlm_client
        self.max_assertions = max_assertions_per_block
        self.prompt_template = _load_prompt_template()
        self._assertion_counter = 0

    def extract(self, block: Block, caption: str = "",
                surrounding_text: str = "") -> List[Assertion]:
        """Extract assertions from a visual block (Table/Figure)."""
        if block.block_type not in (BlockType.TABLE, BlockType.FIGURE):
            return []

        if not block.image_path:
            logger.warning(f"No image crop for visual block {block.id}")
            return []

        prompt = self.prompt_template.replace(
            "{section_path}", block.section_path or "N/A"
        ).replace(
            "{caption}", caption or "N/A"
        ).replace(
            "{ocr_text}", block.text[:500] if block.text else "N/A"
        )

        try:
            response = self.vlm.generate(prompt, image_path=block.image_path)
            raw_assertions = self._parse_response(response)
        except Exception as e:
            logger.warning(f"VLM extraction failed for {block.id}: {e}")
            return []

        assertions = []
        for raw in raw_assertions[:self.max_assertions]:
            assertion = self._to_assertion(raw, block)
            if assertion:
                assertions.append(assertion)

        return assertions

    def extract_batch(self, blocks: List[Block],
                      captions: Dict[str, str] = None) -> Dict[str, List[Assertion]]:
        """Batch extraction from visual blocks."""
        captions = captions or {}
        results = {}

        visual_blocks = [b for b in blocks
                         if b.block_type in (BlockType.TABLE, BlockType.FIGURE)
                         and b.image_path]

        for block in visual_blocks:
            caption = captions.get(block.id, "")
            assertions = self.extract(block, caption=caption)
            results[block.id] = assertions

        return results

    def _parse_response(self, response: str) -> List[dict]:
        response = response.strip()
        start = response.find("[")
        end = response.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            return []

    def _to_assertion(self, raw: dict, block: Block) -> Optional[Assertion]:
        subject = raw.get("subject", "").strip()
        predicate = raw.get("predicate", "").strip()
        obj = raw.get("object", "").strip()

        if not subject or not predicate or not obj:
            return None

        predicate = normalize_predicate(predicate)

        # Stage 1: Self-verification (built into prompt)
        if not raw.get("grounded", True):
            return None

        self._assertion_counter += 1
        assertion_id = f"va_{self._assertion_counter:06d}"

        qualifiers = raw.get("qualifiers", {})
        visual_evidence = raw.get("visual_evidence", "")
        if visual_evidence:
            qualifiers["_visual_evidence"] = visual_evidence

        return Assertion(
            id=assertion_id,
            subject_id=subject,
            predicate=predicate,
            object_id=obj,
            object_is_literal=self._is_literal(obj),
            qualifiers=qualifiers,
            modality=Modality.VISUAL,
            confidence=1.0,
            grounded=True,
            scope_atoms=block.section_path.split(" > ") if block.section_path else [],
            source_support_ids=[block.id],
        )

    @staticmethod
    def _is_literal(value: str) -> bool:
        if any(c.isdigit() for c in value) and len(value) < 50:
            return True
        return False
