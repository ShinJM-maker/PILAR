"""Two-stage quality control for VLM-derived assertions."""
import logging
import re
from typing import List, Dict, Tuple
from src.kg.schema import Assertion, Block, Modality, BlockType

logger = logging.getLogger(__name__)


class QualityController:
    """Two-stage grounding filter for assertions.

    Stage 1: Self-verification (already done in extraction prompt — filter ungrounded).
    Stage 2: Cross-modal span alignment — verify VLM values against OCR text.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def filter_assertions(self, assertions: List[Assertion],
                          blocks: Dict[str, Block]) -> List[Assertion]:
        """Apply quality control to a list of assertions.

        Args:
            assertions: Raw assertions from extractors.
            blocks: Dict of block_id -> Block for cross-modal checking.

        Returns:
            Filtered list of assertions.
        """
        # Stage 1: Already handled — assertions with grounded=False were already filtered
        stage1_passed = [a for a in assertions if a.grounded]
        logger.info(f"Stage 1 (self-verif): {len(assertions)} -> {len(stage1_passed)}")

        # Stage 2: Cross-modal span alignment (only for visual assertions)
        stage2_passed = []
        for assertion in stage1_passed:
            if assertion.modality == Modality.VISUAL:
                if self._cross_modal_check(assertion, blocks):
                    stage2_passed.append(assertion)
                else:
                    logger.debug(f"Stage 2 filter: {assertion.id} failed cross-modal check")
            else:
                stage2_passed.append(assertion)

        logger.info(f"Stage 2 (cross-modal): {len(stage1_passed)} -> {len(stage2_passed)}")
        return stage2_passed

    def _cross_modal_check(self, assertion: Assertion,
                           blocks: Dict[str, Block]) -> bool:
        """Cross-modal span alignment check.

        For VLM assertions:
        - Values from charts/tables must be consistent with OCR text or caption numbers.
        - Diagram entity names must appear in OCR text or governing section.
        """
        if not self.strict_mode:
            return True

        # Get the source blocks' OCR text
        source_texts = []
        for sid in assertion.source_support_ids:
            block = blocks.get(sid)
            if block and block.text:
                source_texts.append(block.text.lower())

        if not source_texts:
            # No OCR text to verify against — pass with lower confidence
            assertion.confidence *= 0.8
            return True

        combined_text = " ".join(source_texts)

        # Check if subject entity name appears in OCR or nearby text
        subject_lower = assertion.subject_id.lower()
        subject_found = self._fuzzy_match(subject_lower, combined_text)

        # Check if object appears (for non-literals, check entity name;
        # for literals like numbers, check the value)
        object_lower = assertion.object_id.lower()
        object_found = self._fuzzy_match(object_lower, combined_text)

        # At least one of subject/object should be verifiable
        if subject_found or object_found:
            return True

        # Check nearby blocks (siblings in same section)
        # This is a relaxed check — if we can't verify at all, lower confidence
        assertion.confidence *= 0.6
        return not self.strict_mode

    @staticmethod
    def _fuzzy_match(needle: str, haystack: str, min_overlap: float = 0.5) -> bool:
        """Check if needle appears in haystack with some fuzzy tolerance."""
        if not needle or not haystack:
            return False

        # Exact substring match
        if needle in haystack:
            return True

        # Word-level overlap
        needle_words = set(needle.split())
        haystack_words = set(haystack.split())
        if not needle_words:
            return False

        overlap = len(needle_words & haystack_words) / len(needle_words)
        return overlap >= min_overlap

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        """Extract numeric values from text."""
        return re.findall(r'\d+\.?\d*', text)
