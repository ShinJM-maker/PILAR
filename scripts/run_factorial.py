"""Full factorial experiment: M(multiplier) x L(local) x G(gate) = 8 conditions.

Usage:
    PYTHONPATH=. python scripts/run_factorial.py
"""
import json
import logging
import time
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

# 8 conditions: (M, L, G)
CONDITIONS = [
    ("E000", 0, 0, 0),
    ("E100", 1, 0, 0),
    ("E010", 0, 1, 0),
    ("E001", 0, 0, 1),
    ("E110", 1, 1, 0),
    ("E101", 1, 0, 1),
    ("E011", 0, 1, 1),
    ("E111", 1, 1, 1),
]

DATASET = "m3docvqa"
AGENT = "naive_rag"
LIMIT = 500
CONCURRENT = 16
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"


def make_config(base_config: dict, M: int, L: int, G: int) -> dict:
    """Override retrieval config for factorial condition."""
    config = deepcopy(base_config)
    r = config["retrieval"]
    r["initial_multiplier"] = 3 if M else 2
    r["enable_local_bonus"] = bool(L)
    r["enable_expanded_gate"] = bool(G)
    return config


def run_one(config: dict, cond_id: str):
    """Run a single factorial condition."""
    from run_experiment import (
        load_dataset, setup_backend, setup_agent,
        _process_sample,
    )
    from src.evaluation.metrics import evaluate_qa
    from src.reader.prompt_templates import format_qa_prompt
    import asyncio
    import random

    logger.info(f"=== {cond_id}: agent={AGENT} backend=vega_kg dataset={DATASET} ===")
    logger.info(f"  M={config['retrieval'].get('initial_multiplier', 2)} "
                f"L={config['retrieval'].get('enable_local_bonus')} "
                f"G={config['retrieval'].get('enable_expanded_gate')}")

    samples = load_dataset(config, DATASET)
    if LIMIT and LIMIT < len(samples):
        random.seed(42)
        samples = random.sample(samples, LIMIT)

    retriever, reader = setup_backend(config, "vega_kg", DATASET)
    agent = setup_agent(config, AGENT, retriever, reader)

    start_time = time.time()

    # 2-phase pipeline (same as run_experiment naive_rag path)
    vlm_client = reader.vlm
    results = [None] * len(samples)

    # Phase 1: Retrieve
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

    # Phase 2: Batch VLM
    logger.info(f"Phase 2: Batch VLM calls...")
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
                        model=vlm_client.model,
                        messages=messages,
                        temperature=vlm_client.temperature,
                        max_tokens=vlm_client.max_tokens,
                    )
                    from src.reader.llm_client import _strip_think_tags
                    return _strip_think_tags(response.choices[0].message.content)
                except Exception as e:
                    logger.warning(f"VLM call failed: {e}")
                    return ""

        tasks = [_call_vlm(p) for p in all_prompts]
        return await asyncio.gather(*tasks)

    answers = asyncio.run(_batch_vlm())
    t_vlm = time.time() - t1
    logger.info(f"Phase 2 done: {t_vlm:.0f}s")

    # Phase 3: Assemble results
    for idx, s in enumerate(samples):
        gold = str(s["answer"]) if s["answer"] is not None else ""
        results[idx] = {
            "id": s.get("id", ""),
            "question": s["question"],
            "gold": gold,
            "prediction": answers[idx].strip() if answers[idx] else "",
        }

    elapsed = time.time() - start_time

    predictions = [r["prediction"] for r in results]
    golds = [r["gold"] for r in results]

    metrics = evaluate_qa(predictions, golds,
                          anls_threshold=config["evaluation"]["anls_threshold"])
    metrics["time_seconds"] = elapsed
    metrics["num_samples"] = len(samples)
    metrics["condition"] = cond_id
    metrics["M"] = config["retrieval"].get("initial_multiplier", 2)
    metrics["L"] = int(config["retrieval"].get("enable_local_bonus", False))
    metrics["G"] = int(config["retrieval"].get("enable_expanded_gate", False))

    logger.info(f"{cond_id}: EM={metrics['EM']:.4f} ANLS={metrics['ANLS']:.4f} "
                f"ROUGE-L={metrics['ROUGE-L']:.4f}")

    # Save
    output_dir = Path(config["paths"]["results"]) / "factorial"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / f"metrics_{cond_id}.json")
    save_json(results, output_dir / f"predictions_{cond_id}.json")

    return metrics


def main():
    base_config = load_config("config/default.yaml")
    base_config["models"]["text_embedder"] = EMBEDDER

    all_metrics = []
    for cond_id, M, L, G in CONDITIONS:
        config = make_config(base_config, M, L, G)
        try:
            metrics = run_one(config, cond_id)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"FAILED {cond_id}: {e}")
            import traceback
            traceback.print_exc()
            all_metrics.append({"condition": cond_id, "error": str(e)})

    # Summary table
    output_dir = Path(base_config["paths"]["results"]) / "factorial"
    save_json(all_metrics, output_dir / "factorial_summary.json")

    print("\n" + "=" * 80)
    print("FACTORIAL RESULTS")
    print("=" * 80)
    print(f"{'ID':<6} {'M':>2} {'L':>2} {'G':>2} {'EM':>8} {'ANLS':>8} {'ROUGE-L':>8}")
    print("-" * 50)
    for m in all_metrics:
        if "error" in m:
            print(f"{m['condition']:<6} -- ERROR: {m['error'][:40]}")
        else:
            print(f"{m['condition']:<6} {m['M']:>2} {m['L']:>2} {m['G']:>2} "
                  f"{m['EM']:>8.4f} {m['ANLS']:>8.4f} {m['ROUGE-L']:>8.4f}")

    # Compute main effects
    if len([m for m in all_metrics if "error" not in m]) == 8:
        print("\n" + "=" * 80)
        print("MAIN EFFECTS (EM)")
        print("=" * 80)
        by_cond = {m["condition"]: m for m in all_metrics}

        def avg_em(*ids):
            return sum(by_cond[i]["EM"] for i in ids) / len(ids)

        eff_M = avg_em("E100", "E110", "E101", "E111") - avg_em("E000", "E010", "E001", "E011")
        eff_L = avg_em("E010", "E110", "E011", "E111") - avg_em("E000", "E100", "E001", "E101")
        eff_G = avg_em("E001", "E101", "E011", "E111") - avg_em("E000", "E100", "E010", "E110")

        print(f"  Effect(M) = {eff_M:+.4f}  (initial_multiplier 2→3)")
        print(f"  Effect(L) = {eff_L:+.4f}  (local bonus)")
        print(f"  Effect(G) = {eff_G:+.4f}  (expanded gate)")

        # 2-way interactions
        print("\n2-WAY INTERACTIONS (EM)")
        print("-" * 50)
        ml_11 = by_cond["E110"]["EM"]; ml_10 = by_cond["E100"]["EM"]
        ml_01 = by_cond["E010"]["EM"]; ml_00 = by_cond["E000"]["EM"]
        int_ML = (ml_11 - ml_10) - (ml_01 - ml_00)

        mg_11 = by_cond["E101"]["EM"]; mg_10 = by_cond["E100"]["EM"]
        mg_01 = by_cond["E001"]["EM"]; mg_00 = by_cond["E000"]["EM"]
        int_MG = (mg_11 - mg_10) - (mg_01 - mg_00)

        lg_11 = by_cond["E011"]["EM"]; lg_10 = by_cond["E010"]["EM"]
        lg_01 = by_cond["E001"]["EM"]; lg_00 = by_cond["E000"]["EM"]
        int_LG = (lg_11 - lg_10) - (lg_01 - lg_00)

        print(f"  M×L = {int_ML:+.4f}")
        print(f"  M×G = {int_MG:+.4f}")
        print(f"  L×G = {int_LG:+.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
