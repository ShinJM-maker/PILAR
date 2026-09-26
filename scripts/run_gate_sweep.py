"""Gate threshold sweep. Fixed: M=0, L=adjacent only (0.04, sec=0).

Usage:
    PYTHONPATH=. python scripts/run_gate_sweep.py
"""
import json
import logging
import time
import random
import asyncio
from pathlib import Path
from copy import deepcopy

from src.utils.io_utils import load_config, load_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

DATASET = "m3docvqa"
AGENT = "naive_rag"
LIMIT = 500
CONCURRENT = 16
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"

# Gate threshold sweep (epsilon fixed at 0.03)
# Then epsilon sweep at best threshold
CONDITIONS = [
    # threshold sweep
    ("G1", 0.25, 0.03),
    ("G2", 0.30, 0.03),
    ("G3", 0.35, 0.03),  # current
    ("G4", 0.40, 0.03),
    # epsilon sweep at 0.35 (will also cover other thresholds if needed)
    ("Eps1", 0.35, 0.00),
    ("Eps3", 0.35, 0.05),
]


def make_config(base_config, threshold, epsilon):
    config = deepcopy(base_config)
    r = config["retrieval"]
    r["initial_multiplier"] = 2
    r["enable_local_bonus"] = True
    r["enable_expanded_gate"] = True
    r["adjacent_page_bonus"] = 0.04
    r["same_section_bonus"] = 0.00
    r["local_bonus_cap"] = 0.05
    r["expanded_dense_threshold"] = threshold
    r["expanded_accept_epsilon"] = epsilon
    return config


def run_one(config, cond_id):
    from run_experiment import load_dataset, setup_backend, setup_agent
    from src.evaluation.metrics import evaluate_qa
    from src.reader.prompt_templates import format_qa_prompt

    r = config["retrieval"]
    logger.info(f"=== {cond_id}: threshold={r['expanded_dense_threshold']} "
                f"epsilon={r['expanded_accept_epsilon']} ===")

    samples = load_dataset(config, DATASET)
    if LIMIT and LIMIT < len(samples):
        random.seed(42)
        samples = random.sample(samples, LIMIT)

    retriever, reader = setup_backend(config, "vega_kg", DATASET)
    agent = setup_agent(config, AGENT, retriever, reader)

    vlm_client = reader.vlm
    start_time = time.time()

    # Phase 1: Retrieve
    logger.info(f"Phase 1: Retrieving {len(samples)} samples...")
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
            logger.info(f"  Retrieval: {idx+1}/{len(samples)}")
    t_ret = time.time() - t0
    logger.info(f"Phase 1 done: {t_ret:.0f}s")

    # Phase 2: Batch VLM
    logger.info(f"Phase 2: Batch VLM...")
    t1 = time.time()

    async def _batch_vlm():
        sem = asyncio.Semaphore(CONCURRENT)
        async_client = vlm_client._get_async_client()

        async def _call_vlm(prompt_text):
            if prompt_text is None:
                return ""
            async with sem:
                try:
                    messages = [{"role": "user", "content": prompt_text}]
                    response = await async_client.chat.completions.create(
                        model=vlm_client.model, messages=messages,
                        temperature=vlm_client.temperature,
                        max_tokens=vlm_client.max_tokens,
                    )
                    from src.reader.llm_client import _strip_think_tags
                    return _strip_think_tags(response.choices[0].message.content)
                except Exception as e:
                    return ""

        return await asyncio.gather(*[_call_vlm(p) for p in all_prompts])

    answers = asyncio.run(_batch_vlm())
    t_vlm = time.time() - t1
    logger.info(f"Phase 2 done: {t_vlm:.0f}s")

    results = []
    for idx, s in enumerate(samples):
        gold = str(s["answer"]) if s["answer"] is not None else ""
        results.append({
            "id": s.get("id", ""),
            "question": s["question"],
            "gold": gold,
            "prediction": answers[idx].strip() if answers[idx] else "",
        })

    elapsed = time.time() - start_time
    predictions = [r["prediction"] for r in results]
    golds = [r["gold"] for r in results]

    metrics = evaluate_qa(predictions, golds,
                          anls_threshold=config["evaluation"]["anls_threshold"])
    metrics["time_seconds"] = elapsed
    metrics["num_samples"] = len(samples)
    metrics["condition"] = cond_id
    metrics["expanded_dense_threshold"] = r["expanded_dense_threshold"]
    metrics["expanded_accept_epsilon"] = r["expanded_accept_epsilon"]

    logger.info(f"{cond_id}: EM={metrics['EM']:.4f} ANLS={metrics['ANLS']:.4f} "
                f"ROUGE-L={metrics['ROUGE-L']:.4f}")

    output_dir = Path(config["paths"]["results"]) / "gate_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / f"metrics_{cond_id}.json")
    save_json(results, output_dir / f"predictions_{cond_id}.json")

    return metrics


def main():
    base_config = load_config("config/default.yaml")
    base_config["models"]["text_embedder"] = EMBEDDER

    all_metrics = []
    for cond_id, threshold, epsilon in CONDITIONS:
        config = make_config(base_config, threshold, epsilon)
        try:
            metrics = run_one(config, cond_id)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"FAILED {cond_id}: {e}")
            import traceback
            traceback.print_exc()
            all_metrics.append({"condition": cond_id, "error": str(e)})

    output_dir = Path(base_config["paths"]["results"]) / "gate_sweep"
    save_json(all_metrics, output_dir / "gate_sweep_summary.json")

    print("\n" + "=" * 70)
    print("GATE SWEEP RESULTS")
    print("=" * 70)
    print(f"{'ID':<6} {'thresh':>7} {'eps':>6} {'EM':>8} {'ANLS':>8} {'ROUGE-L':>8}")
    print("-" * 55)
    for m in all_metrics:
        if "error" in m:
            print(f"{m['condition']:<6} ERROR: {m['error'][:40]}")
        else:
            print(f"{m['condition']:<6} {m['expanded_dense_threshold']:>7.2f} "
                  f"{m['expanded_accept_epsilon']:>6.2f} "
                  f"{m['EM']:>8.4f} {m['ANLS']:>8.4f} {m['ROUGE-L']:>8.4f}")


if __name__ == "__main__":
    main()
