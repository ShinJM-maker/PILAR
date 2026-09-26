# PILAR

Code for the paper **"PILAR: A Page-Grounded Unified Evidence Representation via an Entity-Linked Assertion Graph for Open-Domain QA Agents over Multimodal Document Corpora"** (EMNLP 2026).

PILAR represents multimodal document evidence as page-grounded assertion, support, and provenance objects, links them through an entity-linked assertion graph, and uses the graph as a controlled linking and ranking layer on top of a hybrid BM25 + dense page retriever. The resulting evidence packets are consumed by single-shot and multi-step QA agents (Naive RAG, ReAct, PlanRAG, AutoGen).

> **Naming note.** The project was developed under the internal name *VEGA-KG*, and some identifiers still use it. The backend `vega_kg` is **PILAR**, and `vega_page_only` is the page-only retrieval variant used in the ablation.

## Repository layout

```
config/default.yaml     paths, models, retrieval and reader settings
src/preprocess/         PDF rendering, OCR, layout detection, block extraction
src/dhp/                section-path (hierarchy) recovery and document descriptors
src/kg/                 text/visual assertion extraction, entity linking, QC, graph assembly
src/retrieval/          BM25/dense indices, PILAR retriever, graph expansion, packet assembly, baselines
src/reader/             reader client, prompt templates, packet serializer
src/agents/             Naive RAG, ReAct, ReAct-lite, PlanRAG, AutoGen agents
src/evaluation/         EM / ANLS / ROUGE-L / METEOR, retrieval and grounding metrics
run_kg_build.py         build the assertion graph
run_experiment.py       run agent x backend x dataset experiments
scripts/                index building, baseline preprocessing, and paper analyses
```

## Setup

Python 3.10+ and CUDA GPUs are assumed. Tesseract OCR must be installed (`apt install tesseract-ocr`).

```bash
pip install -r requirements.txt
# or: bash scripts/setup_env.sh   (creates ./venv)
```

Models are served through OpenAI-compatible vLLM endpoints:

| Role | Model | Port |
|---|---|---|
| Text LLM (assertion extraction, agent reasoning) | `Qwen/Qwen3-8B` | 8000 |
| VLM (visual assertions) and reader | `Qwen/Qwen3-VL-8B-Instruct` | 8001 |

```bash
bash scripts/start_vllm.sh    # GPU 0 and GPU 1
bash scripts/stop_vllm.sh     # stops all `vllm serve` processes owned by the current user
```

## Data

Benchmark documents are **not redistributed**. Obtain M3DocVQA and Frames from their official sources and place them as follows (paths can be changed in `config/default.yaml`; relative paths are resolved from the repository root):

```
dataset/m3docvqa/pdfs/<doc_id>.pdf
dataset/m3docvqa/samples.json
dataset/frames/pdfs/<doc_id>.pdf
dataset/frames/samples.json
```

Each `samples.json` is a list of objects with `id`, `question`, `answer`, and `supporting_doc_ids` (list of PDF stems). An optional `metadata` field carries query attributes (hop count, modality, question type) used by the attribute analysis.

## Pipeline

All commands are run from the repository root.

```bash
export PYTHONPATH=.

# 1. Preprocess PDFs (rendering, OCR, layout, blocks, section paths)
python -m src.preprocess.run_preprocess --dataset all

# 2. Build the entity-linked assertion graph
python run_kg_build.py --dataset all

# 3. Build page-level BM25 and dense indices (the paper uses Qwen3-Embedding-0.6B)
python scripts/build_indices.py --dataset all
python scripts/build_qwen_dense.py --dataset all

# 4. (Optional) preprocessing for the ColPali and SimpleDoc baselines
#    ColPali requires transformers>=5; run this in a separate environment if needed.
python scripts/precompute_colpali.py --dataset m3docvqa
python scripts/generate_page_descriptions.py --dataset m3docvqa

# 5. Run experiments
python run_experiment.py --dataset m3docvqa --agent naive_rag --backend vega_kg \
    --embedder Qwen/Qwen3-Embedding-0.6B
```

`--agent` accepts `naive_rag`, `react`, `react_lite`, `planrag`, `autogen`, or `all`; `--backend` accepts `flat_chunk`, `ms_graphrag`, `lightrag`, `simpledoc`, `colpali_page`, `multidocfusion`, `vega_kg`, `vega_page_only`, or `all`. Predictions and metrics are written to `data/results/`. `scripts/run_all_experiments.sh` and `scripts/run_parallel_experiments.sh` run the full grid.

## Scripts for paper analyses

| Paper section | Script |
|---|---|
| Main comparison (agent x backend) | `run_experiment.py`, `scripts/run_all_experiments.sh`, `scripts/run_simpledoc_colpali_experiments.sh` |
| Query-attribute localization | `scripts/analyze_by_attributes.py` |
| Strict matched-interface ablation | `scripts/run_strict_ablation.py` |
| Retrieval-stack factorial, local prior, and gate settings | `scripts/run_factorial.py`, `scripts/run_local_sweep.py`, `scripts/run_gate_sweep.py` |
| Transfer of retrieval hyperparameters across agents | `scripts/run_multiplier_sweep.py` |
| Statistical significance tests | `scripts/significance_test.py` |
| Bottleneck, oracle, and pruning diagnostics | `scripts/run_bottleneck_analysis.py`, `scripts/run_selection_experiments.py` |
| Reader-family comparison | `scripts/run_reader_ab.py` |
| Visual answer-source audit and modality oracle | `scripts/run_visual_audit.py`, `scripts/run_visual_oracle.py` |
| Frames support-chain stress test | `scripts/run_frames_recompute.py`, `scripts/frames_deep_analysis.py`, `scripts/frames_hop_analysis_f3.py` |

## Scope of this release

This repository contains the PILAR pipeline and the following backends from the main comparison: Flat chunk, MultiDocFusion, MS GraphRAG, LightRAG, SimpleDoc, and ColPali page retrieval, together with the PILAR page-only variant. The remaining baselines in the paper (LumberChunker, Meta Chunker, structural chunking, RAPTOR, HopRAG, M3DocRAG, VDocRAG, MoLoRAG) were run outside this codebase and are not included here.

## Citation

```bibtex
@inproceedings{shin2026pilar,
  title     = {{PILAR}: A Page-Grounded Unified Evidence Representation via an Entity-Linked Assertion Graph for Open-Domain {QA} Agents over Multimodal Document Corpora},
  author    = {Shin, Joongmin and Shim, Gyuho and Lee, Jung-hun and Seo, Jaehyung},
  year      = {2026},
  note      = {To appear at EMNLP 2026}
}
```

## License

Code is released under the MIT License. Model weights and benchmark data are subject to their own licenses.
