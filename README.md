# Legal Document RAG Pipeline

A local Retrieval Augmented Generation (RAG) pipeline for querying legal contracts and policy PDFs with source-grounded answers and citations. Built to minimize hallucinated responses through confidence scoring and refusal handling.

## Pipeline Overview

```
PDF Ingestion → Chunking → Embedding → Vector Index → Retrieval → Grounded Answer Generation
```

## Features

- **Page-aware PDF ingestion** — every chunk maps back to a source file and page number for accurate citations
- **Overlapping fixed-size chunking** (512 chars, 100 overlap) — reduces the chance of splitting legal clauses across chunk boundaries
- **Semantic embeddings** via `BAAI/bge-small-en-v1.5` — strong English retrieval performance, runs on CPU
- **FAISS vector search** — fast local similarity search with disk persistence, no external infrastructure
- **Confidence scoring** — retrieval similarity determines answer confidence
- **Grounded answers with citations** — responses are generated only from retrieved chunks
- **Refusal handling** — the system declines to answer when retrieved context is insufficient, rather than speculating
- **Evaluation harness** — Precision@3 measured against a manual QA set

## Quickstart

```bash
pip install -r requirements.txt
python evaluate.py ./data/sample_pdfs --index ./index
```

### Output

The evaluation script reports:

- Precision@3 score
- Per-question retrieval results
- Source references (file + page number)

## Evaluation

| Metric | Result |
|---|---|
| Precision@3 | 1.00 (10/10 on evaluation set) |

**Caveat:** the evaluation corpus is small and controlled (10 hand-written QA pairs over sample PDFs). This validates the pipeline end-to-end but is not representative of production retrieval performance on large, heterogeneous legal corpora.

## Key Design Decisions

Full rationale and trade-offs are documented in [DESIGN.md](./DESIGN.md). Highlights:

- **Chunking:** 512-character chunks with 100-character overlap balance retrieval precision against context preservation. Page-awareness preserves citation quality.
- **Embeddings:** `bge-small-en-v1.5` was chosen over larger models — comparable recall on a small corpus with much lower indexing time and memory footprint.
- **Vector store:** FAISS over Chroma (unnecessary abstraction for a local build) and Pinecone (managed infra not needed for local execution).
- **Hallucination mitigation:** answers are generated strictly from retrieved chunks; weak retrieval similarity triggers a refusal instead of a speculative answer. For legal QA, a false refusal is cheaper than a confident hallucination.

## Limitations

- Dense-only retrieval can miss exact keyword-heavy clauses and numeric language
- Conservative refusal threshold may reject some answerable queries
- Evaluation set is small; results should be treated as a smoke test, not a benchmark

## Roadmap / Scaling Path

Planned improvements for larger corpora (50k+ documents):

- **Hybrid retrieval:** BM25 + dense fusion for keyword-heavy legal language
- **Reranking:** cross-encoder reranking stage on top-k candidates
- **Indexing:** IVF / HNSW approximate nearest-neighbor indexes
- **Metadata:** PostgreSQL-backed metadata store with vendor / contract-type filtering
- **Serving:** stateless API service, warm model loading, query caching, request queueing
- **Ingestion:** parallel embedding jobs and incremental background indexing

## Tech Stack

Python · FAISS · sentence-transformers (`BAAI/bge-small-en-v1.5`) · PyPDF-based extraction

## License

MIT
