"""Build the VEGA-KG knowledge graph from preprocessed documents.

Usage:
    python run_kg_build.py --config config/default.yaml --dataset m3docvqa
    python run_kg_build.py --dataset frames --limit 50
    python run_kg_build.py --dataset all --max-blocks-per-doc 50
"""
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm

from src.utils.io_utils import load_config, load_json, save_json
from src.kg.schema import Block, BlockType
from src.kg.graph_builder import HybridKGBuilder
from src.kg.text_assertion import TextAssertionExtractor
from src.kg.visual_assertion import VisualAssertionExtractor
from src.kg.quality_control import QualityController
from src.kg.entity_linker import EntityLinker
from src.kg.predicate_inventory import PredicateNormalizer
from src.dhp.hierarchy_parser import HierarchyParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)


def build_kg(config: dict, dataset_name: str, limit: int = None,
             max_blocks_per_doc: int = 50):
    """Build VEGA-KG for a dataset."""
    preprocessed_dir = Path(config["paths"]["preprocessed"])
    kg_dir = Path(config["paths"]["kg"])
    kg_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint file for resumability
    checkpoint_file = kg_dir / f"{dataset_name}_checkpoint.json"
    processed_docs = set()
    all_assertions_data = []
    if checkpoint_file.exists():
        ckpt = load_json(checkpoint_file)
        processed_docs = set(ckpt.get("processed_docs", []))
        all_assertions_data = ckpt.get("assertions", [])
        logger.info(f"Resuming from checkpoint: {len(processed_docs)} docs already processed")

    # Initialize components
    from src.reader.llm_client import VLLMClient

    text_llm = VLLMClient(
        base_url="http://localhost:8000/v1",
        model=config["models"]["text_llm"],
        max_tokens=1024,
        max_concurrent=32,
    )
    vlm = VLLMClient(
        base_url="http://localhost:8001/v1",
        model=config["models"]["vlm"],
        max_tokens=1024,
        max_concurrent=8,
    )

    text_extractor = TextAssertionExtractor(text_llm, config["kg"]["max_assertions_per_block"])
    visual_extractor = VisualAssertionExtractor(vlm, config["kg"]["max_assertions_per_block"])
    qc = QualityController(strict_mode=True)
    hierarchy_parser = HierarchyParser(max_depth=config["dhp"]["max_depth"])

    # Load preprocessed documents
    doc_files = sorted(preprocessed_dir.glob("*.json"))
    doc_files = [f for f in doc_files if not f.name.startswith("_")]
    if limit:
        doc_files = doc_files[:limit]

    logger.info(f"Building KG from {len(doc_files)} documents ({len(processed_docs)} already done)")

    all_blocks = []
    all_assertions = []

    # Reconstruct previously extracted assertions
    for ad in all_assertions_data:
        from src.kg.schema import Assertion, Modality
        all_assertions.append(Assertion(
            id=ad["id"], subject_id=ad["subject_id"],
            predicate=ad["predicate"], object_id=ad["object_id"],
            object_is_literal=ad.get("object_is_literal", False),
            qualifiers=ad.get("qualifiers", {}),
            modality=Modality(ad.get("modality", "text")),
            confidence=ad.get("confidence", 1.0),
            grounded=ad.get("grounded", True),
            scope_atoms=ad.get("scope_atoms", []),
            source_support_ids=ad.get("source_support_ids", []),
        ))

    save_interval = 50  # Save checkpoint every N docs

    for i, doc_file in enumerate(tqdm(doc_files, desc="Processing documents")):
        doc_id = doc_file.stem
        if doc_id in processed_docs:
            # Still need to load blocks for graph building later
            doc_data = load_json(doc_file)
            blocks = [Block.from_dict(b) for b in doc_data.get("blocks", [])]
            all_blocks.extend(blocks)
            continue

        doc_data = load_json(doc_file)
        blocks = [Block.from_dict(b) for b in doc_data.get("blocks", [])]

        if not blocks:
            processed_docs.add(doc_id)
            continue

        # Hierarchy parsing
        blocks = hierarchy_parser.parse(blocks)
        block_map = {b.id: b for b in blocks}

        # Text assertion extraction (batch with concurrency)
        text_blocks = [b for b in blocks
                       if b.block_type in (BlockType.PARAGRAPH, BlockType.CAPTION,
                                            BlockType.HEADER, BlockType.LIST)
                       and b.text.strip() and len(b.text.strip()) >= 20]
        # Cap blocks per doc
        if len(text_blocks) > max_blocks_per_doc:
            text_blocks = text_blocks[:max_blocks_per_doc]

        if text_blocks:
            text_results = text_extractor.extract_batch(text_blocks)
            for block_id, assertions in text_results.items():
                all_assertions.extend(assertions)

        # Visual assertion extraction (only for blocks with image crops)
        visual_blocks = [b for b in blocks
                         if b.block_type in (BlockType.TABLE, BlockType.FIGURE)
                         and b.image_path]
        for vblock in visual_blocks[:10]:  # Cap visual blocks too
            caption = ""
            for b in blocks:
                if (b.block_type == BlockType.CAPTION and
                    b.page == vblock.page and
                    abs(b.bbox[1] - vblock.bbox[3]) < 50):
                    caption = b.text
                    break
            va = visual_extractor.extract(vblock, caption=caption)
            all_assertions.extend(va)

        all_blocks.extend(blocks)
        processed_docs.add(doc_id)

        # Periodic checkpoint
        if (i + 1) % save_interval == 0:
            _save_checkpoint(checkpoint_file, processed_docs, all_assertions)
            logger.info(f"Checkpoint saved: {len(processed_docs)} docs, {len(all_assertions)} assertions")

    # Final checkpoint
    _save_checkpoint(checkpoint_file, processed_docs, all_assertions)

    logger.info(f"Extracted {len(all_assertions)} raw assertions from {len(all_blocks)} blocks")

    # Quality control
    block_map = {b.id: b for b in all_blocks}
    filtered_assertions = qc.filter_assertions(all_assertions, block_map)
    logger.info(f"After QC: {len(filtered_assertions)} assertions")

    # Entity linking
    entity_linker = EntityLinker(
        link_threshold=config["kg"]["entity_link_threshold"],
        cross_doc_threshold=config["kg"]["cross_doc_threshold"],
    )
    linked_assertions = entity_linker.link_assertions(filtered_assertions, block_map)
    logger.info(f"Entity inventory: {len(entity_linker.entity_inventory)} entities")

    # Predicate normalization
    pred_normalizer = PredicateNormalizer(config["kg"]["predicate_merge_threshold"])
    all_predicates = list(set(a.predicate for a in linked_assertions))
    if all_predicates:
        pred_mapping = pred_normalizer.cluster_predicates(all_predicates)
        for a in linked_assertions:
            if a.predicate in pred_mapping:
                a.predicate = pred_mapping[a.predicate]

    # Build graph
    kg = HybridKGBuilder()
    kg.build_from_blocks(all_blocks, linked_assertions, entity_linker.entity_inventory)

    # Cross-document entity links
    cross_doc_links = entity_linker.add_cross_doc_links()
    for e1, e2 in cross_doc_links:
        kg.add_same_as_edge(e1, e2)
    logger.info(f"Added {len(cross_doc_links)} cross-document entity links")

    # Save
    output_path = kg_dir / f"{dataset_name}_kg.pkl"
    kg.save(str(output_path))
    logger.info(f"KG saved to {output_path}")
    logger.info(f"  Nodes: {kg.graph.number_of_nodes()}, Edges: {kg.graph.number_of_edges()}")

    # Save entity inventory
    entity_data = {eid: e.to_dict() for eid, e in entity_linker.entity_inventory.items()}
    save_json(entity_data, kg_dir / f"{dataset_name}_entities.json")

    # Cleanup checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return kg


def _save_checkpoint(path: Path, processed_docs: set, assertions: list):
    """Save checkpoint for resumability."""
    data = {
        "processed_docs": list(processed_docs),
        "assertions": [
            {
                "id": a.id, "subject_id": a.subject_id,
                "predicate": a.predicate, "object_id": a.object_id,
                "object_is_literal": a.object_is_literal,
                "qualifiers": a.qualifiers,
                "modality": a.modality.value,
                "confidence": a.confidence,
                "grounded": a.grounded,
                "scope_atoms": a.scope_atoms,
                "source_support_ids": a.source_support_ids,
            }
            for a in assertions
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Build VEGA-KG knowledge graph")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", choices=["m3docvqa", "frames", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-blocks-per-doc", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)

    datasets = ["m3docvqa", "frames"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        build_kg(config, dataset, limit=args.limit,
                 max_blocks_per_doc=args.max_blocks_per_doc)


if __name__ == "__main__":
    main()
