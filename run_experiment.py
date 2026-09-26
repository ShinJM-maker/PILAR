"""Main experiment runner for VEGA-KG.

Usage:
    python run_experiment.py --dataset m3docvqa --agent naive_rag --backend flat_chunk --limit 50
    python run_experiment.py --dataset m3docvqa --agent all --backend all
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.utils.io_utils import load_config, load_json, save_json
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

BACKENDS = ["flat_chunk", "ms_graphrag", "lightrag", "simpledoc", "colpali_page", "multidocfusion", "vega_kg", "vega_page_only"]
AGENTS = ["naive_rag", "react", "planrag", "autogen", "react_lite"]
DATASETS = ["m3docvqa", "frames"]

# Cache for loaded chunks and indices (shared across backends)
_CACHE = {}


def load_dataset(config: dict, dataset_name: str) -> List[Dict]:
    """Load QA dataset samples."""
    if dataset_name == "m3docvqa":
        path = config["paths"]["samples_m3docvqa"]
    elif dataset_name == "frames":
        path = config["paths"]["samples_frames"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    samples = load_json(path)
    for s in samples:
        if "supporting_doc_ids" not in s and "filtered_supporting_doc_ids" in s:
            s["supporting_doc_ids"] = s["filtered_supporting_doc_ids"]
        if "supporting_doc_ids" not in s:
            s["supporting_doc_ids"] = []
    logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
    return samples


def _get_dataset_doc_ids(config: dict, dataset_name: str) -> set:
    """Get the set of document IDs referenced by the dataset."""
    samples = load_dataset(config, dataset_name)
    doc_ids = set()
    for s in samples:
        doc_ids.update(s.get("supporting_doc_ids", []))
    return doc_ids


def _load_chunks_and_indices(config: dict, dataset_name: str):
    """Load pre-built chunks and indices (with caching)."""
    embedder_model = config["models"]["text_embedder"]
    embedder_revision = config["models"].get("text_embedder_revision")
    cache_key = f"chunks_{dataset_name}_{embedder_model}_{embedder_revision}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    indices_dir = Path(config["paths"]["indices"])

    # Load pre-built page chunks
    chunks_path = indices_dir / f"{dataset_name}_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Pre-built chunks not found at {chunks_path}. "
            f"Run: PYTHONPATH=. python scripts/build_indices.py --dataset {dataset_name}"
        )
    chunks = load_json(chunks_path)
    logger.info(f"Loaded {len(chunks)} pre-built page chunks")

    # Load pre-built BM25 index
    import pickle
    bm25_path = indices_dir / f"{dataset_name}_bm25.pkl"
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    logger.info(f"Loaded BM25 index from {bm25_path}")

    # Load pre-built dense index (model-specific filename)
    if "qwen3" in embedder_model.lower() or "Qwen3" in embedder_model:
        dense_suffix = "_dense_qwen3.npz"
    else:
        dense_suffix = "_dense.npz"
    dense_path = indices_dir / f"{dataset_name}{dense_suffix}"
    dense = DenseIndex(model_name=embedder_model, revision=embedder_revision)
    if dense_path.exists():
        dense.load_embeddings(str(dense_path))
        logger.info(f"Loaded dense index from {dense_path}")
    else:
        logger.warning(f"No pre-built dense index at {dense_path}, building now...")
        documents = [{"id": cid, "text": c["text"]} for cid, c in chunks.items()]
        dense.build(documents)
        dense.save(str(dense_path))

    result = {"chunks": chunks, "bm25": bm25, "dense": dense}
    _CACHE[cache_key] = result
    return result


def setup_backend(config: dict, backend_name: str, dataset_name: str, samples=None):
    """Initialize retrieval backend and reader."""
    from src.reader.llm_client import VLLMClient
    from src.reader.qwen_reader import QwenReader

    reader_client = VLLMClient(
        base_url="http://localhost:8001/v1",
        model=config["models"]["reader"],
        temperature=config["reader"]["temperature"],
        max_tokens=config["reader"]["max_new_tokens"],
    )
    reader = QwenReader(
        reader_client,
        token_budget=config["reader"]["context_limit"],
        max_images=config["retrieval"]["max_images_per_packet"],
        max_new_tokens=config["reader"]["max_new_tokens"],
    )

    if backend_name == "vega_kg":
        retriever = _setup_vega_kg(config, dataset_name)
    elif backend_name == "vega_page_only":
        retriever = _setup_vega_page_only(config, dataset_name)
    else:
        # All non-KG backends share the same chunks/indices
        data = _load_chunks_and_indices(config, dataset_name)
        chunks, bm25, dense = data["chunks"], data["bm25"], data["dense"]

        if backend_name == "flat_chunk":
            from src.retrieval.retriever import FlatChunkRetriever
            retriever = FlatChunkRetriever(bm25, dense, chunks,
                                            config["retrieval"]["token_budget"])
        elif backend_name == "ms_graphrag":
            from src.retrieval.baseline_backends import MSGraphRAGRetriever
            kg_dir = Path(config["paths"]["kg"])
            entities_path = kg_dir / f"{dataset_name}_entities.json"
            entities = load_json(entities_path) if entities_path.exists() else {}
            retriever = MSGraphRAGRetriever(bm25, dense, chunks, entities,
                                             config["retrieval"]["token_budget"])
        elif backend_name == "lightrag":
            from src.retrieval.baseline_backends import LightRAGRetriever
            kg_dir = Path(config["paths"]["kg"])
            checkpoint = kg_dir / f"{dataset_name}_checkpoint.json"
            assertions = []
            if checkpoint.exists():
                ckpt = load_json(checkpoint)
                assertions = ckpt.get("assertions", [])
            retriever = LightRAGRetriever(bm25, dense, chunks, assertions,
                                           config["retrieval"]["token_budget"])
        elif backend_name in ("simpledoc", "colpali_page"):
            from src.retrieval.baseline_backends import SimpleDocRetriever, ColPaliPrecomputedScores
            indices_dir = Path(config["paths"]["indices"])

            # ColPali pre-computed scores
            colpali_scores_path = indices_dir / f"{dataset_name}_colpali_query_scores.json"
            colpali_index = None
            if colpali_scores_path.exists():
                colpali_index = ColPaliPrecomputedScores(str(colpali_scores_path))
                if samples:
                    colpali_index.set_query_mapping(samples)
                logger.info(f"SimpleDoc: loaded ColPali pre-computed scores")

            if backend_name == "colpali_page":
                # ColPali-only: visual channel only (no text, no description)
                retriever = SimpleDocRetriever(bm25, dense, chunks,
                                                config["retrieval"]["token_budget"],
                                                colpali_index=colpali_index,
                                                alpha_text=0.0, alpha_visual=1.0,
                                                alpha_desc=0.0)
            else:
                # Full SimpleDoc: text + ColPali + description
                desc_path = indices_dir / f"{dataset_name}_page_descriptions.json"
                desc_index_path = indices_dir / f"{dataset_name}_desc_dense.npz"
                descriptions = {}
                desc_index = None
                if desc_path.exists():
                    descriptions = load_json(desc_path)
                    logger.info(f"SimpleDoc: loaded {len(descriptions)} page descriptions")
                if desc_index_path.exists():
                    embedder_name = config["models"].get("text_embedder",
                                                          "sentence-transformers/all-MiniLM-L6-v2")
                    desc_index = DenseIndex(model_name=embedder_name, device="cpu")
                    desc_index.load_embeddings(str(desc_index_path))
                    logger.info(f"SimpleDoc: loaded description dense index")

                retriever = SimpleDocRetriever(bm25, dense, chunks,
                                                config["retrieval"]["token_budget"],
                                                colpali_index=colpali_index,
                                                desc_index=desc_index,
                                                descriptions=descriptions)
        elif backend_name == "multidocfusion":
            from src.retrieval.baseline_backends import MultiDocFusionRetriever
            retriever = MultiDocFusionRetriever(bm25, dense, chunks,
                                                config["retrieval"]["token_budget"])
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    return retriever, reader


def _setup_vega_kg(config: dict, dataset_name: str):
    """Setup VEGA-KG hybrid retriever (page-level retrieval + KG expansion)."""
    from src.kg.graph_builder import HybridKGBuilder
    from src.retrieval.retriever import VEGAKGRetriever

    kg_path = Path(config["paths"]["kg"]) / f"{dataset_name}_kg.pkl"
    if not kg_path.exists():
        raise FileNotFoundError(
            f"KG not found at {kg_path}. Run `python run_kg_build.py --dataset {dataset_name}` first."
        )

    kg = HybridKGBuilder.load(str(kg_path))
    logger.info(f"Loaded KG: {len(kg.supports)} supports, {len(kg.assertions)} assertions, "
                f"{len(kg.entities)} entities")

    # Load page-level chunks and indices (same as flat_chunk backend)
    data = _load_chunks_and_indices(config, dataset_name)
    page_chunks, page_bm25, page_dense = data["chunks"], data["bm25"], data["dense"]

    # Convert entities and assertions to dicts for the retriever
    entities_dict = {eid: e.to_dict() for eid, e in kg.entities.items()}
    assertions_dict = {aid: a.to_dict() for aid, a in kg.assertions.items()}

    return VEGAKGRetriever(
        kg.graph, kg.supports,
        page_bm25, page_dense, page_chunks,
        entities_dict, assertions_dict,
        config["retrieval"],
    )


def _setup_vega_page_only(config: dict, dataset_name: str):
    """Setup VEGA page-only baseline: same packing as vega_kg v1 but NO KG expansion.
    This is an internal ablation baseline to isolate KG expansion effect.
    """
    from src.retrieval.retriever import VEGAPageOnlyRetriever

    data = _load_chunks_and_indices(config, dataset_name)
    page_chunks, page_bm25, page_dense = data["chunks"], data["bm25"], data["dense"]

    return VEGAPageOnlyRetriever(
        page_bm25, page_dense, page_chunks,
        config["retrieval"]["token_budget"],
        initial_multiplier=config["retrieval"].get("initial_multiplier", 3),
    )


def setup_agent(config: dict, agent_name: str, retriever, reader):
    """Initialize the agent framework."""
    if agent_name == "naive_rag":
        from src.agents.naive_rag import NaiveRAGAgent
        return NaiveRAGAgent(retriever, reader, top_k=config["retrieval"]["seed_top_k"])

    from src.reader.llm_client import VLLMClient
    reasoning_llm = VLLMClient(
        base_url="http://localhost:8000/v1",
        model=config["models"]["text_llm"],
        max_tokens=512,
    )

    if agent_name == "react":
        from src.agents.react_agent import ReActAgent
        return ReActAgent(retriever, reader, reasoning_llm,
                         max_steps=config["agents"]["react_max_steps"],
                         detailed_logging=config.get("detailed_logging", False),
                         prompt_horizon=config["agents"].get("react_prompt_horizon"),
                         log_naive_gate=config["agents"].get("react_log_naive_gate", True))
    elif agent_name == "planrag":
        from src.agents.planrag_agent import PlanRAGAgent
        return PlanRAGAgent(retriever, reader, reasoning_llm,
                           max_subqueries=config["agents"]["planrag_max_subqueries"])
    elif agent_name == "react_lite":
        from src.agents.react_lite_agent import ReActLiteAgent
        return ReActLiteAgent(
            retriever, reader, reasoning_llm,
            max_steps=config["agents"].get("react_lite_max_steps", 2),
            enable_naive_gate=config.get("enable_naive_gate", False),
            enable_condition_verifier=config.get("enable_condition_verifier", False),
            enable_deny_router=config.get("enable_deny_router", False),
            detailed_logging=config.get("detailed_logging", False),
        )
    elif agent_name == "autogen":
        from src.agents.autogen_agent import AutoGenAgent
        return AutoGenAgent(retriever, reader, reasoning_llm,
                           max_turns=config["agents"]["autogen_max_turns"])
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def _process_sample(agent, sample):
    """Process a single sample (for concurrent execution)."""
    question = sample["question"]
    gold_answer = str(sample["answer"]) if sample["answer"] is not None else ""
    try:
        result = agent.answer(question)
        pred_answer = result["answer"]
    except Exception as e:
        logger.error(f"Error on '{question[:50]}...': {e}")
        pred_answer = ""
        result = {"answer": "", "evidence": [], "metadata": {"error": str(e)}}

    output = {
        "id": sample.get("id", ""),
        "question": question,
        "gold": gold_answer,
        "prediction": pred_answer,
        "metadata": result.get("metadata", {}),
    }
    # Save evidence when detailed logging is enabled (needed for replay)
    if result.get("metadata", {}).get("step_logs") is not None:
        output["evidence"] = result.get("evidence", [])
    return output


def run_experiment(config: dict, dataset_name: str, agent_name: str,
                   backend_name: str, limit: int = None,
                   concurrent: int = 1) -> Dict:
    """Run a single experiment: agent x backend x dataset."""
    logger.info(f"=== Experiment: {agent_name} x {backend_name} x {dataset_name} ===")

    samples = load_dataset(config, dataset_name)
    if limit:
        if limit < len(samples):
            import random
            random.seed(42)
            samples = random.sample(samples, limit)
        else:
            samples = samples[:limit]

    retriever, reader = setup_backend(config, backend_name, dataset_name, samples=samples)
    agent = setup_agent(config, agent_name, retriever, reader)

    start_time = time.time()
    results = []

    if agent_name == "naive_rag" and concurrent > 1:
        # Pipeline mode: pre-retrieve all evidence (CPU), then batch VLM calls
        # This avoids GIL contention between retrieval and VLM I/O
        from src.reader.prompt_templates import format_qa_prompt
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        results = [None] * len(samples)
        vlm_client = reader.vlm

        # Phase 1: Retrieve all evidence sequentially (CPU-bound)
        logger.info(f"Phase 1: Retrieving evidence for {len(samples)} samples...")
        t0 = time.time()
        all_prompts = []
        for idx, s in enumerate(samples):
            try:
                evidence = agent.retriever.retrieve(s["question"], top_k=agent.top_k)
                context = reader.serializer.serialize(evidence, None)
                prompt = format_qa_prompt(question=s["question"], context=context)
                all_prompts.append(prompt)
            except Exception as e:
                logger.error(f"Retrieval error [{idx}]: {e}")
                all_prompts.append(None)
            if (idx + 1) % 200 == 0:
                logger.info(f"Retrieval: {idx+1}/{len(samples)} "
                           f"({(time.time()-t0)/(idx+1):.2f}s/sample)")
        t_ret = time.time() - t0
        logger.info(f"Phase 1 done: {t_ret:.0f}s ({t_ret/len(samples):.2f}s/sample)")

        # Phase 2: Batch async VLM calls
        logger.info(f"Phase 2: Batch VLM calls for {len(samples)} samples...")
        t1 = time.time()

        async def _batch_vlm():
            sem = asyncio.Semaphore(concurrent)
            async_client = vlm_client._get_async_client()

            async def _call_vlm(prompt_text):
                if prompt_text is None:
                    return ""
                async with sem:
                    try:
                        messages = [{"role": "user", "content": prompt_text}]
                        response = await async_client.chat.completions.create(
                            model=vlm_client.model,
                            messages=messages,
                            temperature=vlm_client.temperature,
                            max_tokens=vlm_client.max_tokens,
                        )
                        from src.reader.llm_client import _strip_think_tags
                        return _strip_think_tags(
                            response.choices[0].message.content)
                    except Exception as e:
                        logger.warning(f"VLM call failed: {e}")
                        return ""

            tasks = [_call_vlm(p) for p in all_prompts]
            return await asyncio.gather(*tasks)

        answers = asyncio.run(_batch_vlm())
        t_vlm = time.time() - t1
        logger.info(f"Phase 2 done: {t_vlm:.0f}s ({t_vlm/len(samples):.2f}s/sample)")

        # Phase 3: Assemble results
        for idx, s in enumerate(samples):
            gold = str(s["answer"]) if s["answer"] is not None else ""
            results[idx] = {
                "id": s.get("id", ""),
                "question": s["question"],
                "gold": gold,
                "prediction": answers[idx].strip() if answers[idx] else "",
                "metadata": {"agent": "naive_rag"},
            }
        logger.info(f"{agent_name}/{backend_name}: {len(samples)}/{len(samples)} done "
                    f"(retrieval={t_ret:.0f}s, vlm={t_vlm:.0f}s)")

    elif concurrent > 1:
        # Use thread pool for multi-step agents
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = {executor.submit(_process_sample, agent, s): i
                       for i, s in enumerate(samples)}
            results = [None] * len(samples)
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                done += 1
                if done % 100 == 0:
                    logger.info(f"{agent_name}/{backend_name}: {done}/{len(samples)}")
        logger.info(f"{agent_name}/{backend_name}: {len(samples)}/{len(samples)} done")
    else:
        for sample in tqdm(samples, desc=f"{agent_name}/{backend_name}"):
            results.append(_process_sample(agent, sample))

    predictions = [r["prediction"] for r in results]
    golds = [r["gold"] for r in results]

    elapsed = time.time() - start_time

    from src.evaluation.metrics import evaluate_qa
    metrics = evaluate_qa(predictions, golds,
                          anls_threshold=config["evaluation"]["anls_threshold"])
    metrics["time_seconds"] = elapsed
    metrics["num_samples"] = len(samples)
    # Add embedder tag if non-default
    embedder = config["models"]["text_embedder"]
    embedder_tag = ""
    if "qwen3" in embedder.lower() or "Qwen3" in embedder:
        embedder_tag = "_qwen3emb"
    # Build react_lite variant tag
    variant_tag = ""
    if agent_name == "react_lite":
        parts = []
        if config.get("enable_naive_gate"):
            parts.append("gated")
        if config.get("enable_condition_verifier"):
            parts.append("condverifier")
        if config.get("enable_deny_router"):
            parts.append("denyrouter")
        if len(parts) == 3:
            variant_tag = "_full"  # all three = "full"
        elif parts:
            variant_tag = "_" + "_".join(parts)
    metrics["experiment"] = f"{agent_name}{variant_tag}_{backend_name}_{dataset_name}{embedder_tag}"
    metrics["embedder"] = embedder

    logger.info(f"Results: EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}, "
                f"ROUGE-L={metrics['ROUGE-L']:.4f}, METEOR={metrics['METEOR']:.4f}")

    output_dir = Path(config["paths"]["results"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / f"metrics_{metrics['experiment']}.json")
    save_json(results, output_dir / f"predictions_{metrics['experiment']}.json")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run VEGA-KG experiments")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--agent", choices=AGENTS + ["all"], default="all")
    parser.add_argument("--backend", choices=BACKENDS + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrent", type=int, default=8,
                        help="Concurrent samples for multi-step agents")
    parser.add_argument("--embedder", type=str, default=None,
                        help="Override dense embedder model (e.g. Qwen/Qwen3-Embedding-0.6B)")
    parser.add_argument("--no-shutdown", action="store_true",
                        help="Don't shutdown vLLM after experiment")
    parser.add_argument("--detailed-logging", action="store_true",
                        help="Enable step-level detailed logging (ReAct)")
    parser.add_argument("--enable-naive-gate", action="store_true",
                        help="Enable supported-naive gate (react_lite)")
    parser.add_argument("--enable-condition-verifier", action="store_true",
                        help="Enable condition-only verifier (react_lite)")
    parser.add_argument("--enable-deny-router", action="store_true",
                        help="Enable deny-list router (react_lite)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.embedder:
        config["models"]["text_embedder"] = args.embedder
        logger.info(f"Using embedder: {args.embedder}")
    if args.detailed_logging:
        config["detailed_logging"] = True
        logger.info("Detailed step-level logging enabled")
    if args.enable_naive_gate:
        config["enable_naive_gate"] = True
        logger.info("Supported-naive gate enabled")
    if args.enable_condition_verifier:
        config["enable_condition_verifier"] = True
        logger.info("Condition-only verifier enabled")
    if args.enable_deny_router:
        config["enable_deny_router"] = True
        logger.info("Deny-list router enabled")

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    agents = AGENTS if args.agent == "all" else [args.agent]
    backends = BACKENDS if args.backend == "all" else [args.backend]

    all_metrics = []
    for dataset in datasets:
        for backend in backends:
            for agent in agents:
                try:
                    metrics = run_experiment(config, dataset, agent, backend,
                                             args.limit, args.concurrent)
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"FAILED: {agent}x{backend}x{dataset}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_metrics.append({
                        "experiment": f"{agent}_{backend}_{dataset}",
                        "error": str(e),
                    })

    output_dir = Path(config["paths"]["results"])
    save_json(all_metrics, output_dir / "all_metrics_summary.json")
    logger.info(f"All experiments complete. Results in {output_dir}")


if __name__ == "__main__":
    import sys
    no_shutdown = "--no-shutdown" in sys.argv
    main()
    if not no_shutdown:
        from src.utils.io_utils import shutdown_vllm
        shutdown_vllm()
