"""Local bonus decomposition: adjacent vs same-section vs both.
Fixed: M=0 (multiplier=2), G=1 (expanded gate on).

Usage:
    PYTHONPATH=. python scripts/run_local_sweep.py
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

# Local bonus decomposition conditions
# All: M=0 (multiplier=2), G=1 (gate on)
CONDITIONS = [
    ("L0", 0.00, 0.00, 0.00),   # local off (= E001)
    ("L1", 0.04, 0.00, 0.05),   # adjacent only
    ("L2", 0.00, 0.015, 0.05),  # same-section only
    ("L3", 0.04, 0.015, 0.05),  # both (= E011)
]


def make_config(base_config, adj_bonus, sec_bonus, cap):
    config = deepcopy(base_config)
    r = config["retrieval"]
    r["initial_multiplier"] = 2
    r["enable_local_bonus"] = (adj_bonus > 0 or sec_bonus > 0)
    r["enable_expanded_gate"] = True
    r["adjacent_page_bonus"] = adj_bonus
    r["same_section_bonus"] = sec_bonus
    r["local_bonus_cap"] = cap
    return config


def run_one(config, cond_id):
    from run_experiment import load_dataset, setup_backend, setup_agent
    from src.evaluation.metrics import evaluate_qa
    from src.reader.prompt_templates import format_qa_prompt

    r = config["retrieval"]
    logger.info(f"=== {cond_id}: adj={r['adjacent_page_bonus']} "
                f"sec={r['same_section_bonus']} cap={r['local_bonus_cap']} ===")

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

    # Assemble
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
    metrics["adjacent_page_bonus"] = r["adjacent_page_bonus"]
    metrics["same_section_bonus"] = r["same_section_bonus"]
    metrics["local_bonus_cap"] = r["local_bonus_cap"]

    logger.info(f"{cond_id}: EM={metrics['EM']:.4f} ANLS={metrics['ANLS']:.4f} "
                f"ROUGE-L={metrics['ROUGE-L']:.4f}")

    output_dir = Path(config["paths"]["results"]) / "local_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / f"metrics_{cond_id}.json")
    save_json(results, output_dir / f"predictions_{cond_id}.json")

    return metrics


def main():
    base_config = load_config("config/default.yaml")
    base_config["models"]["text_embedder"] = EMBEDDER

    all_metrics = []
    for cond_id, adj, sec, cap in CONDITIONS:
        config = make_config(base_config, adj, sec, cap)
        try:
            metrics = run_one(config, cond_id)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"FAILED {cond_id}: {e}")
            import traceback
            traceback.print_exc()
            all_metrics.append({"condition": cond_id, "error": str(e)})

    output_dir = Path(base_config["paths"]["results"]) / "local_sweep"
    save_json(all_metrics, output_dir / "local_sweep_summary.json")

    print("\n" + "=" * 70)
    print("LOCAL BONUS DECOMPOSITION")
    print("=" * 70)
    print(f"{'ID':<5} {'adj':>6} {'sec':>6} {'cap':>6} {'EM':>8} {'ANLS':>8} {'ROUGE-L':>8}")
    print("-" * 55)
    for m in all_metrics:
        if "error" in m:
            print(f"{m['condition']:<5} ERROR: {m['error'][:40]}")
        else:
            print(f"{m['condition']:<5} {m['adjacent_page_bonus']:>6.3f} "
                  f"{m['same_section_bonus']:>6.4f} {m['local_bonus_cap']:>6.3f} "
                  f"{m['EM']:>8.4f} {m['ANLS']:>8.4f} {m['ROUGE-L']:>8.4f}")

    # Effects
    valid = {m["condition"]: m for m in all_metrics if "error" not in m}
    if len(valid) == 4:
        print("\nEFFECTS (EM)")
        adj_eff = (valid["L1"]["EM"] + valid["L3"]["EM"]) / 2 - (valid["L0"]["EM"] + valid["L2"]["EM"]) / 2
        sec_eff = (valid["L2"]["EM"] + valid["L3"]["EM"]) / 2 - (valid["L0"]["EM"] + valid["L1"]["EM"]) / 2
        int_as = (valid["L3"]["EM"] - valid["L1"]["EM"]) - (valid["L2"]["EM"] - valid["L0"]["EM"])
        print(f"  Adjacent effect = {adj_eff:+.4f}")
        print(f"  Section effect  = {sec_eff:+.4f}")
        print(f"  Adj×Sec interaction = {int_as:+.4f}")


if __name__ == "__main__":
    main()
