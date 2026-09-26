"""Small 2x2 sweep: initial_multiplier × same_section_bonus for react & autogen.

Tests whether E011 transfer failure is driven by multiplier (pool size) or same_section.
"""
import json
import copy
import sys
sys.path.insert(0, ".")

from src.utils.io_utils import load_config, save_json
from pathlib import Path

CONDITIONS = [
    # (label, initial_multiplier, same_section_bonus)
    ("M2_SS0",   2, 0.00),   # E011 — already have react=0.314, autogen=0.326
    ("M2_SS15",  2, 0.015),  # multiplier=2, default same_section
    ("M3_SS0",   3, 0.00),   # default multiplier, no same_section
    ("M3_SS15",  3, 0.015),  # full default — already have react=0.314, autogen=0.330
]

# Only need to run the two missing conditions
MISSING = ["M2_SS15", "M3_SS0"]
AGENTS = ["react", "autogen"]

def main():
    from run_experiment import run_experiment

    base_config = load_config("config/default.yaml")
    base_config["models"]["text_embedder"] = "Qwen/Qwen3-Embedding-0.6B"

    results_dir = Path(base_config["paths"]["results"]) / "multiplier_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Add existing results
    all_results["react_M2_SS0"] = {"EM": 0.314, "ANLS": 0.3447, "ROUGE-L": 0.3742, "source": "E011 run"}
    all_results["react_M3_SS15"] = {"EM": 0.314, "ANLS": 0.3430, "ROUGE-L": 0.3732, "source": "default run"}
    all_results["autogen_M2_SS0"] = {"EM": 0.326, "ANLS": 0.3533, "ROUGE-L": 0.3845, "source": "E011 run"}
    all_results["autogen_M3_SS15"] = {"EM": 0.330, "ANLS": 0.3573, "ROUGE-L": 0.3875, "source": "default run"}

    for label, mult, ss in CONDITIONS:
        if label not in MISSING:
            continue

        for agent in AGENTS:
            key = f"{agent}_{label}"
            print(f"\n{'='*60}")
            print(f"Running: {key} (multiplier={mult}, same_section={ss})")
            print(f"{'='*60}")

            config = copy.deepcopy(base_config)
            config["retrieval"]["initial_multiplier"] = mult
            config["retrieval"]["same_section_bonus"] = ss
            # Keep other E011 params fixed
            config["retrieval"]["adjacent_page_bonus"] = 0.04
            config["retrieval"]["expanded_dense_threshold"] = 0.35
            config["retrieval"]["expanded_accept_epsilon"] = 0.03

            metrics = run_experiment(config, "m3docvqa", agent, "vega_kg", limit=500, concurrent=8)

            all_results[key] = {
                "EM": metrics["EM"],
                "ANLS": metrics["ANLS"],
                "ROUGE-L": metrics["ROUGE-L"],
                "METEOR": metrics["METEOR"],
                "initial_multiplier": mult,
                "same_section_bonus": ss,
            }

            save_json(metrics, results_dir / f"metrics_{key}.json")
            print(f"  → EM={metrics['EM']:.4f}, ANLS={metrics['ANLS']:.4f}")

    # Save combined summary
    save_json(all_results, results_dir / "sweep_summary.json")

    # Print 2x2 table
    print("\n" + "="*70)
    print("2x2 RESULTS: initial_multiplier × same_section_bonus")
    print("="*70)
    for agent in AGENTS:
        print(f"\n{agent}:")
        print(f"  {'':20s} ss=0.00    ss=0.015")
        for mult in [2, 3]:
            ss0_key = f"{agent}_M{mult}_SS0"
            ss15_key = f"{agent}_M{mult}_SS15"
            em0 = all_results[ss0_key]["EM"]
            em15 = all_results[ss15_key]["EM"]
            print(f"  multiplier={mult}:     {em0:.3f}      {em15:.3f}")

    # Main effect analysis
    print("\n" + "="*70)
    print("MAIN EFFECTS (EM)")
    print("="*70)
    for agent in AGENTS:
        m2 = (all_results[f"{agent}_M2_SS0"]["EM"] + all_results[f"{agent}_M2_SS15"]["EM"]) / 2
        m3 = (all_results[f"{agent}_M3_SS0"]["EM"] + all_results[f"{agent}_M3_SS15"]["EM"]) / 2
        ss0 = (all_results[f"{agent}_M2_SS0"]["EM"] + all_results[f"{agent}_M3_SS0"]["EM"]) / 2
        ss15 = (all_results[f"{agent}_M2_SS15"]["EM"] + all_results[f"{agent}_M3_SS15"]["EM"]) / 2
        print(f"\n{agent}:")
        print(f"  multiplier effect: M2={m2:.3f} vs M3={m3:.3f} → delta={m2-m3:+.3f}")
        print(f"  same_section effect: SS0={ss0:.3f} vs SS15={ss15:.3f} → delta={ss0-ss15:+.3f}")


if __name__ == "__main__":
    main()
