"""Visual Evidence Oracle — determine optimal input modality for visual queries.

Tests 4 conditions on visual-classified queries (n≈140):
  V-O0: current final context (text-only retrieval) — reused from reader_ab R0
  V-O1: gold page text only (O1-identical) — reused from reader_ab R2
  V-O2: gold page IMAGE only (pure visual ceiling)
  V-O3: gold page IMAGE + OCR/text (multimodal ceiling)

Reuses reader_ab predictions.json for V-O0/V-O1; runs new VLM calls for V-O2/V-O3.

Usage:
    PYTHONPATH=. python scripts/run_visual_oracle.py
"""
import argparse
import asyncio
import base64
import io
import json
import logging
import random
import re
import time
from pathlib import Path

import fitz  # PyMuPDF

from src.utils.io_utils import load_config, load_json, save_json
from src.evaluation.metrics import evaluate_qa, normalize_answer
from src.reader.llm_client import _strip_think_tags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DATASET = "m3docvqa"
LIMIT = 500
CONCURRENT = 8  # lower for multimodal (images are larger)
DPI = 150  # good enough for VLM, much smaller than 300

# Visual query keywords (same as reader_ab)
VISUAL_KEYWORDS = [
    "wearing", "worn", "wear", "dressed",
    "color", "colour",
    "facial hair", "beard", "moustache", "mustache",
    "what is shown", "what does it show", "what can be seen",
    "how many people", "how many persons",
    "what type of", "what kind of",
    "logo", "sign", "icon", "picture", "image", "photo", "photograph",
    "chart", "graph", "diagram", "figure",
    "looks like", "look like", "appearance",
    "holding", "held", "carries", "carrying",
    "sitting", "standing", "posing", "pointing",
    "background", "foreground",
    "left", "right", "top", "bottom",
    "wrist", "hand", "eye", "head", "hair", "face",
]
VISUAL_PATTERNS = [
    r"what is \w+ wearing",
    r"what is \w+ holding",
    r"what is on \w+ (head|face|wrist|hand|eyes|neck)",
    r"what position",
    r"what pose",
    r"what gesture",
    r"who is (taller|shorter|bigger|smaller)",
    r"how does .+ look",
]


def classify_query(question: str) -> str:
    q_lower = question.lower()
    cues = 0
    for kw in VISUAL_KEYWORDS:
        if kw in q_lower:
            cues += 1
    for pat in VISUAL_PATTERNS:
        if re.search(pat, q_lower):
            cues += 1
    words = question.split()
    if len(words) <= 8 and any(w.lower().startswith("what") for w in words[:2]):
        if any(w[0].isupper() for w in words[2:] if len(w) > 1):
            cues += 1
    return "visual" if cues >= 1 else "text"


def find_gold_pages(sample, page_chunks, norm_cache):
    """O1-identical: top-level supporting_doc_ids filter."""
    gold = str(sample.get("answer", ""))
    if not gold:
        return []
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        return []
    supporting_docs = sample.get("supporting_doc_ids", [])
    matches = []
    for cid, chunk in page_chunks.items():
        doc_id = chunk.get("doc_id", "")
        if supporting_docs and doc_id not in supporting_docs:
            continue
        text_norm = norm_cache.get(cid, "")
        if text_norm and gold_norm in text_norm:
            matches.append(cid)
    return matches


def estimate_tokens(text):
    return int(len(text.split()) * 1.3)


def build_oracle_context(page_chunks, chunk_ids, token_budget=4500):
    """O1-identical: raw text concatenation within budget."""
    parts = []
    total_tokens = 0
    for cid in chunk_ids:
        chunk = page_chunks.get(cid)
        if not chunk:
            continue
        text = chunk.get("text", "")
        if not text:
            continue
        est = estimate_tokens(text)
        if total_tokens + est > token_budget:
            break
        parts.append(text)
        total_tokens += est
    return "\n\n".join(parts)


def parse_chunk_id(chunk_id: str):
    """Parse '{doc_id}_p{page:04d}' → (doc_id, page_num)."""
    # Last part is _pXXXX
    idx = chunk_id.rfind("_p")
    if idx == -1:
        return chunk_id, 0
    doc_id = chunk_id[:idx]
    page_str = chunk_id[idx + 2:]
    try:
        page_num = int(page_str)
    except ValueError:
        page_num = 0
    return doc_id, page_num


def render_page_to_bytes(pdf_path: str, page_num: int, dpi: int = 150) -> bytes:
    """Render a single PDF page to PNG bytes."""
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        return None
    page = doc[page_num]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def make_image_prompt(question: str) -> str:
    """Prompt for image-only condition (V-O2)."""
    return (
        "/no_think\n"
        "You are an AI assistant that answers questions by analyzing provided documents.\n"
        "Answer the [Question] using only the image above.\n"
        "- Give a SHORT, DIRECT answer (a few words or a short phrase).\n"
        "- Do NOT explain your reasoning. Do NOT include citations.\n"
        "- If unsure, output \"unanswerable\".\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Answer]"
    )


def make_image_text_prompt(question: str, context: str) -> str:
    """Prompt for image+text condition (V-O3)."""
    return (
        "/no_think\n"
        "You are an AI assistant that answers questions by analyzing provided documents.\n"
        "Answer the [Question] using both the image above AND the [Context] below.\n"
        "- Give a SHORT, DIRECT answer (a few words or a short phrase).\n"
        "- Do NOT explain your reasoning. Do NOT include citations.\n"
        "- If unsure, output \"unanswerable\".\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Context]\n"
        f"{context}\n\n"
        "[Answer]"
    )


def batch_multimodal_vlm_call(base_url, model, items, concurrent=8,
                               temperature=0.0, max_tokens=1024):
    """Batch async multimodal VLM calls.

    items: list of dicts with keys:
        - prompt: str (text prompt)
        - image_b64: str or None (base64-encoded PNG)
    Returns list of response strings.
    """
    async def _batch():
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        sem = asyncio.Semaphore(concurrent)

        async def _call(item):
            if item is None or item.get("prompt") is None:
                return ""
            async with sem:
                try:
                    content = []
                    if item.get("image_b64"):
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{item['image_b64']}"
                            },
                        })
                    content.append({"type": "text", "text": item["prompt"]})
                    messages = [{"role": "user", "content": content}]
                    response = await client.chat.completions.create(
                        model=model, messages=messages,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    return _strip_think_tags(response.choices[0].message.content)
                except Exception as e:
                    logger.error(f"VLM error: {e}")
                    return ""

        return await asyncio.gather(*[_call(it) for it in items])

    return asyncio.run(_batch())


def main():
    config = load_config("config/default.yaml")
    pdf_dir = Path(config["paths"]["pdf_m3docvqa"])

    from run_experiment import load_dataset, setup_backend

    samples = load_dataset(config, DATASET)
    random.seed(42)
    if LIMIT and LIMIT < len(samples):
        samples = random.sample(samples, LIMIT)
    logger.info(f"Loaded {len(samples)} samples")

    # Setup retriever to get page_chunks
    from copy import deepcopy
    exp_config = deepcopy(config)
    exp_config["models"]["text_embedder"] = "Qwen/Qwen3-Embedding-0.6B"
    r = exp_config["retrieval"]
    r["initial_multiplier"] = 2
    r["enable_local_bonus"] = True
    r["enable_expanded_gate"] = True
    r["adjacent_page_bonus"] = 0.04
    r["same_section_bonus"] = 0.00
    r["local_bonus_cap"] = 0.05
    r["expanded_dense_threshold"] = 0.35
    r["expanded_accept_epsilon"] = 0.03

    retriever, reader = setup_backend(exp_config, "vega_kg", DATASET)
    page_chunks = retriever.page_chunks

    current_url = reader.vlm.base_url
    current_model = reader.vlm.model
    logger.info(f"Reader: {current_model} @ {current_url}")

    output_dir = Path(config["paths"]["results"]) / "visual_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build norm cache
    logger.info("Building norm cache...")
    t0 = time.time()
    norm_cache = {}
    for cid, chunk in page_chunks.items():
        t = chunk.get("text", "")
        if t:
            norm_cache[cid] = normalize_answer(t)
    logger.info(f"  Cache: {len(norm_cache)} entries ({time.time()-t0:.0f}s)")

    # ============================================================
    # PHASE 1: Classify + find gold pages for all 500 samples
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Classify + find gold pages")
    logger.info("=" * 60)

    all_data = []
    for idx, s in enumerate(samples):
        q = s["question"]
        gold = str(s["answer"]) if s["answer"] is not None else ""
        query_type = classify_query(q)
        gold_page_ids = find_gold_pages(s, page_chunks, norm_cache)

        all_data.append({
            "idx": idx,
            "id": s.get("id", ""),
            "question": q,
            "gold": gold,
            "query_type": query_type,
            "gold_page_ids": gold_page_ids,
            "gold_in_index": len(gold_page_ids) > 0,
        })

    visual_data = [d for d in all_data if d["query_type"] == "visual"]
    visual_with_gold = [d for d in visual_data if d["gold_in_index"]]
    logger.info(f"Total: {len(all_data)}, Visual: {len(visual_data)}, "
                f"Visual with gold: {len(visual_with_gold)}")

    # ============================================================
    # PHASE 2: Load V-O0 and V-O1 from reader_ab predictions
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Reuse V-O0 (R0) and V-O1 (R2) from reader_ab")
    logger.info("=" * 60)

    reader_ab_preds_path = Path(config["paths"]["results"]) / "reader_ab" / "predictions.json"
    reader_ab_preds = load_json(reader_ab_preds_path)

    # Map by id
    ab_by_id = {p["id"]: p for p in reader_ab_preds}

    for d in all_data:
        ab = ab_by_id.get(d["id"], {})
        d["pred_VO0"] = ab.get("pred_R0", "")
        d["pred_VO1"] = ab.get("pred_R2", "")

    # Verify V-O0 and V-O1 match reader_ab
    vo0_visual = [d for d in visual_data]
    golds_v = [d["gold"] for d in vo0_visual]
    preds_vo0 = [d["pred_VO0"] for d in vo0_visual]
    preds_vo1 = [d["pred_VO1"] for d in vo0_visual]
    m_vo0 = evaluate_qa(preds_vo0, golds_v, anls_threshold=config["evaluation"]["anls_threshold"])
    m_vo1 = evaluate_qa(preds_vo1, golds_v, anls_threshold=config["evaluation"]["anls_threshold"])
    logger.info(f"  V-O0 (visual, n={len(vo0_visual)}): EM={m_vo0['EM']:.4f}")
    logger.info(f"  V-O1 (visual, n={len(vo0_visual)}): EM={m_vo1['EM']:.4f}")

    # ============================================================
    # PHASE 3: Render gold page images
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Render gold page images for visual queries")
    logger.info("=" * 60)

    t0 = time.time()
    render_count = 0
    render_errors = 0

    for d in visual_data:
        d["gold_image_b64"] = None
        d["gold_image_doc_id"] = None
        d["gold_image_page"] = None

        if not d["gold_page_ids"]:
            continue

        # Use first gold page
        first_gold = d["gold_page_ids"][0]
        doc_id, page_num = parse_chunk_id(first_gold)
        pdf_path = pdf_dir / f"{doc_id}.pdf"

        if not pdf_path.exists():
            render_errors += 1
            continue

        try:
            png_bytes = render_page_to_bytes(str(pdf_path), page_num, dpi=DPI)
            if png_bytes:
                d["gold_image_b64"] = base64.b64encode(png_bytes).decode("utf-8")
                d["gold_image_doc_id"] = doc_id
                d["gold_image_page"] = page_num
                render_count += 1
        except Exception as e:
            logger.warning(f"Render error for {first_gold}: {e}")
            render_errors += 1

    t1 = time.time()
    logger.info(f"  Rendered {render_count} pages, {render_errors} errors ({t1-t0:.0f}s)")

    # Also build gold text for V-O3
    token_budget = exp_config["retrieval"].get("token_budget", 4500)
    for d in visual_data:
        if d["gold_page_ids"]:
            d["gold_text"] = build_oracle_context(page_chunks, d["gold_page_ids"],
                                                   token_budget=token_budget)
        else:
            d["gold_text"] = ""

    # ============================================================
    # PHASE 4: V-O2 — gold page IMAGE only
    # ============================================================
    logger.info("=" * 60)
    logger.info("V-O2: gold page IMAGE only (pure visual ceiling)")
    logger.info("=" * 60)

    items_vo2 = []
    for d in visual_data:
        if d["gold_image_b64"]:
            items_vo2.append({
                "prompt": make_image_prompt(d["question"]),
                "image_b64": d["gold_image_b64"],
            })
        else:
            items_vo2.append(None)

    t_start = time.time()
    answers_vo2 = batch_multimodal_vlm_call(
        current_url, current_model, items_vo2, concurrent=CONCURRENT
    )
    t_vo2 = time.time() - t_start
    logger.info(f"  VLM done: {t_vo2:.0f}s")

    for i, d in enumerate(visual_data):
        d["pred_VO2"] = answers_vo2[i].strip() if answers_vo2[i] else ""

    preds_vo2 = [d["pred_VO2"] for d in visual_data]
    m_vo2 = evaluate_qa(preds_vo2, golds_v, anls_threshold=config["evaluation"]["anls_threshold"])
    logger.info(f"  V-O2 (visual, n={len(visual_data)}): EM={m_vo2['EM']:.4f}")

    # ============================================================
    # PHASE 5: V-O3 — gold page IMAGE + text
    # ============================================================
    logger.info("=" * 60)
    logger.info("V-O3: gold page IMAGE + OCR/text (multimodal ceiling)")
    logger.info("=" * 60)

    items_vo3 = []
    for d in visual_data:
        if d["gold_image_b64"] and d["gold_text"]:
            items_vo3.append({
                "prompt": make_image_text_prompt(d["question"], d["gold_text"]),
                "image_b64": d["gold_image_b64"],
            })
        elif d["gold_text"]:
            # No image available, fallback to text-only (same as V-O1)
            from src.reader.prompt_templates import format_qa_prompt
            items_vo3.append({
                "prompt": format_qa_prompt(question=d["question"], context=d["gold_text"]),
                "image_b64": None,
            })
        else:
            items_vo3.append(None)

    t_start = time.time()
    answers_vo3 = batch_multimodal_vlm_call(
        current_url, current_model, items_vo3, concurrent=CONCURRENT
    )
    t_vo3 = time.time() - t_start
    logger.info(f"  VLM done: {t_vo3:.0f}s")

    for i, d in enumerate(visual_data):
        d["pred_VO3"] = answers_vo3[i].strip() if answers_vo3[i] else ""

    preds_vo3 = [d["pred_VO3"] for d in visual_data]
    m_vo3 = evaluate_qa(preds_vo3, golds_v, anls_threshold=config["evaluation"]["anls_threshold"])
    logger.info(f"  V-O3 (visual, n={len(visual_data)}): EM={m_vo3['EM']:.4f}")

    # ============================================================
    # PHASE 6: Conditional — only samples with gold page + image
    # ============================================================
    logger.info("=" * 60)
    logger.info("Conditional: visual queries with gold page AND image rendered")
    logger.info("=" * 60)

    has_image = [d for d in visual_data if d["gold_image_b64"]]
    if has_image:
        g_cond = [d["gold"] for d in has_image]
        for cond, key in [("V-O0", "pred_VO0"), ("V-O1", "pred_VO1"),
                          ("V-O2", "pred_VO2"), ("V-O3", "pred_VO3")]:
            p_cond = [d[key] for d in has_image]
            m_cond = evaluate_qa(p_cond, g_cond,
                                  anls_threshold=config["evaluation"]["anls_threshold"])
            logger.info(f"  {cond} (n={len(has_image)}): EM={m_cond['EM']:.4f} "
                         f"ANLS={m_cond['ANLS']:.4f} ROUGE-L={m_cond['ROUGE-L']:.4f}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("VISUAL EVIDENCE ORACLE RESULTS")
    print("=" * 70)

    results = {
        "V-O0": {"desc": "current final context", **m_vo0, "n": len(visual_data)},
        "V-O1": {"desc": "gold page text only", **m_vo1, "n": len(visual_data)},
        "V-O2": {"desc": "gold page IMAGE only", **m_vo2, "n": len(visual_data)},
        "V-O3": {"desc": "gold page IMAGE + text", **m_vo3, "n": len(visual_data)},
    }

    print(f"\n{'Cond':<6} {'Description':<28} {'n':>4} {'EM':>8} {'ANLS':>8} {'ROUGE-L':>8}")
    print("-" * 70)
    for cond, m in results.items():
        print(f"{cond:<6} {m['desc']:<28} {m['n']:>4} "
              f"{m['EM']:>8.4f} {m['ANLS']:>8.4f} {m['ROUGE-L']:>8.4f}")

    print("\nDELTAS (vs V-O0 baseline)")
    print("-" * 50)
    for cond in ["V-O1", "V-O2", "V-O3"]:
        delta_em = results[cond]["EM"] - results["V-O0"]["EM"]
        delta_anls = results[cond]["ANLS"] - results["V-O0"]["ANLS"]
        print(f"  {cond}-V-O0: EM {delta_em:+.4f}, ANLS {delta_anls:+.4f}")

    print("\nKEY COMPARISONS")
    print("-" * 50)
    d_o2_o1 = results["V-O2"]["EM"] - results["V-O1"]["EM"]
    d_o3_o2 = results["V-O3"]["EM"] - results["V-O2"]["EM"]
    d_o3_o1 = results["V-O3"]["EM"] - results["V-O1"]["EM"]
    print(f"  V-O2 vs V-O1 (image vs text):     {d_o2_o1:+.4f}  "
          f"{'→ IMAGE path needed' if d_o2_o1 > 0.02 else '→ text sufficient'}")
    print(f"  V-O3 vs V-O2 (image+text vs image):{d_o3_o2:+.4f}  "
          f"{'→ multimodal > image-only' if d_o3_o2 > 0.02 else '→ image alone sufficient'}")
    print(f"  V-O3 vs V-O1 (multimodal ceiling): {d_o3_o1:+.4f}  "
          f"{'→ significant multimodal gain' if d_o3_o1 > 0.05 else '→ modest gain'}")

    # Projected overall impact
    print("\nPROJECTED OVERALL EM IMPACT")
    print("-" * 50)
    n_text = sum(1 for d in all_data if d["query_type"] == "text")
    n_vis = len(visual_data)
    text_em = m_vo0["EM"]  # unchanged (using all visual data EM as proxy)
    # Use R0 text EM from reader_ab
    text_em_r0 = 0.400  # known from reader_ab
    for cond in ["V-O1", "V-O2", "V-O3"]:
        vis_em = results[cond]["EM"]
        overall = (text_em_r0 * n_text + vis_em * n_vis) / (n_text + n_vis)
        delta = overall - 0.336  # vs current overall R0
        print(f"  If visual→{cond}: overall EM ≈ {overall:.3f} ({delta:+.3f})")

    # Decision interpretation
    print("\nDECISION RULES")
    print("-" * 50)
    if d_o2_o1 > 0.05:
        print("  ✓ V-O2 >> V-O1: Visual bottleneck IS image evidence path absence")
        print("    → Priority: image-aware inference path (before stronger reader)")
    elif d_o2_o1 > 0.02:
        print("  ~ V-O2 > V-O1: Image helps moderately")
        print("    → Worth adding image path, but not the only bottleneck")
    else:
        print("  ✗ V-O2 ≈ V-O1: Visual queries are mostly solvable from text")
        print("    → Focus on retrieval/reader improvements, not image path")

    if d_o3_o2 > 0.02:
        print("  ✓ V-O3 > V-O2: Multimodal (image+text) is best")
        print("    → Use both modalities in evidence path")
    else:
        print("  ~ V-O3 ≈ V-O2: Text doesn't add much beyond image")

    # Save results
    save_json(results, output_dir / "oracle_metrics.json")

    # Save per-sample predictions (visual only)
    per_sample = []
    for d in visual_data:
        per_sample.append({
            "idx": d["idx"], "id": d["id"],
            "question": d["question"], "gold": d["gold"],
            "gold_page_ids": d["gold_page_ids"],
            "gold_in_index": d["gold_in_index"],
            "has_image": d["gold_image_b64"] is not None,
            "gold_image_doc_id": d.get("gold_image_doc_id"),
            "gold_image_page": d.get("gold_image_page"),
            "pred_VO0": d.get("pred_VO0", ""),
            "pred_VO1": d.get("pred_VO1", ""),
            "pred_VO2": d.get("pred_VO2", ""),
            "pred_VO3": d.get("pred_VO3", ""),
        })
    save_json(per_sample, output_dir / "predictions_visual.json")

    # Save conditional (image-available) metrics
    if has_image:
        cond_results = {}
        g_cond = [d["gold"] for d in has_image]
        for cond, key in [("V-O0", "pred_VO0"), ("V-O1", "pred_VO1"),
                          ("V-O2", "pred_VO2"), ("V-O3", "pred_VO3")]:
            p_cond = [d[key] for d in has_image]
            m_cond = evaluate_qa(p_cond, g_cond,
                                  anls_threshold=config["evaluation"]["anls_threshold"])
            m_cond["n"] = len(has_image)
            cond_results[cond] = m_cond
        save_json(cond_results, output_dir / "oracle_metrics_conditional.json")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
