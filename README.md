# Legal Document RAG Pipeline

A local Retrieval Augmented Generation (RAG) pipeline for querying legal contracts and policy PDFs. Answers are generated strictly from retrieved passages, carry inline citations to source file and page, and the system refuses rather than speculates when the documents don't support an answer.

## Pipeline Overview

```
PDF Ingestion → Chunking → Embedding → Vector Index → Retrieval → Context-Constrained Generation
```

## Features

- **Page-aware PDF ingestion** — every chunk maps back to a source file and page number for accurate citations
- **Overlapping fixed-size chunking** (512 chars, 100 overlap) — reduces the chance of splitting legal clauses across chunk boundaries
- **Semantic embeddings** via `BAAI/bge-small-en-v1.5` — strong English retrieval performance, runs on CPU
- **FAISS vector search** — fast local similarity search with disk persistence, no external infrastructure
- **Context-constrained generation** — answers are produced by an LLM restricted to the retrieved passages via a strict system prompt at `temperature=0`, with inline passage citations and verbatim quoting where exact contract wording matters
- **Two-layer refusal** — a retrieval-similarity gate before the LLM call catches *nothing relevant retrieved*; a model-level `NOT_IN_CONTEXT` signal after generation catches *retrieved but doesn't actually answer*
- **Evaluation harness** — Precision@3 measured against a manual QA set

## Quickstart

```bash
pip install -r requirements.txt

# free key, no card required: https://console.groq.com
export GROQ_API_KEY=your_key_here

python demo_query.py
python evaluate.py ./data/sample_pdfs --index ./index
```

On Windows PowerShell, set the key with `$env:GROQ_API_KEY = "your_key_here"`.

The index is built on first run from the PDFs in `data/sample_pdfs/` and persisted to disk. Delete `index.faiss` and `index.meta` to force a rebuild.

### Example

```
Question: What are the payment terms under the service contract?

Confidence: 0.6727

The payment terms are as follows:
"Undisputed invoices are payable within thirty (30) calendar days from the invoice date" [1, 2].
Late payments may accrue interest at "1.5% per month or the maximum rate permitted by law,
whichever is lower" [2].

Sources:
- contract.pdf | page 1
- contract.pdf | page 1
- contract.pdf | page 2
```

## Evaluation

| Metric | Result |
|---|---|
| Precision@3 | 1.00 (10/10 on evaluation set) |

**Read the caveats before quoting that number.** The corpus is three synthetic documents generated for demonstration — plain text, no letterheads, no scans, no multi-column layouts or tables. The ten questions were written by the same person who read those documents, which biases them toward the phrasing in the text. And only retrieval is measured: there is no evaluation of generated answer quality, faithfulness, or citation correctness. A perfect score here shows the pipeline is wired correctly end to end. It is a smoke test, not a benchmark.

## Key Design Decisions

Full rationale and trade-offs are in [DESIGN.md](./DESIGN.md). Highlights:

- **Chunking:** 512-character chunks with 100-character overlap balance retrieval precision against context preservation. Character-based splitting is naive and can cut mid-clause — sentence- or structure-aware splitting is the natural upgrade.
- **Embeddings:** `bge-small-en-v1.5` over larger models — comparable recall on a small corpus at much lower indexing time and memory cost.
- **Vector store:** FAISS over Chroma (unnecessary abstraction here) and Pinecone (managed infrastructure not needed for a local pipeline).
- **Two-layer refusal:** retrieval similarity is a proxy for relevance, not a measurement of it. A chunk can be topically close to a question, score well, and still contain none of the information needed to answer it. One threshold can't separate those failure modes, so the system checks both before and after generation. For legal QA, a false refusal is cheaper than a confident hallucination.

## Limitations

- Dense-only retrieval can miss exact keyword-heavy clauses and numeric language — precisely the content that matters most in contracts
- The 0.30 confidence threshold was chosen empirically, not calibrated against a labelled answerable/unanswerable set
- No evaluation of generated answer quality, faithfulness, or citation accuracy — only retrieval is measured
- Untested against real-world PDF noise: scanned pages, multi-column layouts, tables, headers bleeding into extraction
- Answer quality depends on the served model

## Roadmap

- **Hybrid retrieval:** BM25 + dense fusion for keyword-heavy legal language
- **Reranking:** cross-encoder stage over top-k candidates
- **Threshold calibration:** sweep against a labelled set, choose the operating point from the answer/refusal trade-off curve
- **Generation evaluation:** faithfulness scoring and citation-accuracy checking
- **Indexing at scale:** IVF / HNSW approximate nearest-neighbour indexes
- **Metadata:** PostgreSQL-backed store with vendor / contract-type filtering
- **Serving:** stateless API, warm model loading, query caching, request queueing

## Tech Stack

Python · FAISS · sentence-transformers (`BAAI/bge-small-en-v1.5`) · PyMuPDF · Groq (Llama)

## License

MIT