# DESIGN.md — RAG Pipeline Architecture

## Overview

This project implements a local Retrieval Augmented Generation (RAG) pipeline for querying legal contracts and policy PDFs. The objective is to answer precise document questions with source citations while minimizing hallucinated responses.

Pipeline flow:

```
Document Ingestion → Chunking → Embedding → Vector Index → Retrieval → Context-Constrained Generation
```

---

## 1. Chunking Strategy

**Chosen approach**

- page-aware text extraction from PDFs
- fixed-size overlapping character chunks
- chunk size: 512 characters
- overlap: 100 characters

**Why**

Legal documents contain clauses that span multiple sentences or continue across paragraph boundaries. Overlap reduces the chance that important language is split between chunks.

Page-awareness preserves citation quality: each retrieved chunk maps directly to a source file, page number, and chunk text.

**Trade-off**

Smaller chunks improve retrieval precision but lose surrounding context. Larger chunks preserve context but reduce granularity. 512/100 balances the two.

**Known weakness**

Character-based splitting is naive — it can cut mid-word or mid-clause, and it ignores document structure entirely. Recursive or sentence-aware splitting, or a structure-aware splitter that respects clause and section numbering, would be the natural upgrade for real contract text.

---

## 2. Embedding Model Choice

**Chosen model:** `BAAI/bge-small-en-v1.5`

**Why**

- strong semantic retrieval performance for English text
- lightweight enough for local CPU execution
- faster indexing than larger transformer embeddings
- low setup overhead

**Trade-off**

Larger embedding models may improve recall, but increase indexing time and memory use without measurable benefit on a corpus this small. This choice was not benchmarked against alternatives — on a larger corpus it should be.

---

## 3. Vector Store Choice

**Chosen store:** FAISS

**Why FAISS**

- fast similarity search
- lightweight local dependency, no external infrastructure
- easy persistence to disk

**Why not Chroma:** convenient for prototyping, but adds abstraction this implementation does not need.

**Why not Pinecone:** managed search is useful in production, but unnecessary for a local pipeline.

---

## 4. Retrieval Strategy

**Implemented**

1. embed the user query
2. search top-k nearest vectors in FAISS (k = 3)
3. return highest-scoring chunks with metadata

**Why**

For a moderate local corpus, dense top-k retrieval is simple, reliable, and fast.

**Trade-off**

Dense retrieval can miss exact keyword-heavy clauses and numeric language — precisely the content that matters most in contracts (dollar amounts, notice periods, defined terms). This is the largest known gap in the current design.

**Planned upgrades**

- BM25 + dense hybrid retrieval
- metadata filtering by vendor / contract type
- cross-encoder reranking
- query expansion for legal terminology

---

## 5. Hallucination Mitigation

This is the core design concern. Legal question answering requires factual precision; a confident wrong answer is worse than no answer.

### Two-layer refusal

**Layer 1 — retrieval-score gate (pre-generation)**

Confidence is computed as the mean similarity score across the top-k retrieved chunks. If confidence falls below a threshold (0.30), the pipeline refuses without calling the LLM at all.

Catches: *nothing relevant exists in the corpus.*

**Layer 2 — model-level refusal (post-generation)**

The generation prompt instructs the model to reply with the sentinel `NOT_IN_CONTEXT` when the supplied passages do not answer the question. The pipeline converts that sentinel into the standard refusal response.

Catches: *chunks scored acceptably but do not actually contain the answer.*

**Why two layers**

Retrieval similarity is a proxy for relevance, not a measurement of it. A chunk can be topically close to a question and score well while containing none of the information needed to answer it. A single score threshold cannot separate those two failure modes, so the system checks both before and after generation.

### Generation constraints

- answers are generated strictly from the retrieved passages, supplied as numbered context
- the system prompt forbids outside knowledge
- the model is instructed to cite the passage number for every claim
- `temperature=0` for determinism and reproducibility

**Trade-off**

A conservative threshold will reject some answerable queries. For legal QA, a false refusal is cheaper than a confident hallucination.

**Known weakness**

The 0.30 threshold was chosen empirically, not calibrated. Proper calibration would sweep the threshold against a labeled set of answerable and unanswerable questions and select the operating point from the resulting answer/refusal trade-off curve.

---

## 6. Evaluation Harness

**Method**

10 manual question–answer pairs written against the sample PDFs.

**Metric:** Precision@3 — how often the correct supporting chunk appears in the top three retrieved results.

**Observed result:** Precision@3 = 1.00 (10/10)

**Limitations — read this before quoting the number**

- The corpus is 3 synthetic documents, generated for demonstration. Plain text, no letterheads, no scans, no multi-column layouts, no tables. This is close to the easiest possible retrieval setting.
- The evaluation questions were written by the same person who saw the documents, which biases them toward the phrasing that appears in the text.
- 10 questions is far too few for the score to be meaningful. A perfect result here indicates the pipeline is wired correctly end to end — it is a smoke test, not a benchmark.
- **Only retrieval is measured.** There is no evaluation of generated answer quality, faithfulness, or citation correctness. That is the significant gap in the current harness.

**What a real evaluation would add**

- faithfulness / groundedness scoring of generated answers against retrieved context
- citation accuracy checking
- an explicit unanswerable-question set to measure refusal behaviour
- Recall@k and MRR alongside Precision@k
- a corpus of real contracts with realistic formatting noise

---

## 7. Scaling to 50,000 Documents

**A. Ingestion / embedding throughput**
Parallel embedding jobs, incremental indexing, background ingestion workers.

**B. Search performance**
Exact nearest-neighbour search degrades with corpus size. Move to approximate search — IVF for partition-based recall/speed trade-offs, HNSW for better recall at higher memory cost.

**C. Metadata storage**
Flat local metadata files stop being manageable. PostgreSQL for production, SQLite for smaller deployments.

**D. Query throughput**
Stateless API service, warm model loading, request queueing, caching of frequent queries.

**E. Retrieval quality**
More documents produce more near-matches. Reranking, metadata filters, and hybrid lexical+dense retrieval all become necessary rather than optional.

---

## 8. Design Philosophy

The implementation prioritises local reproducibility, measurable evaluation, simple architecture, citation accuracy, and conservative answer behaviour.

A clear and reliable baseline was favoured over unnecessary complexity. The known gaps — hybrid retrieval, threshold calibration, and generation-quality evaluation — are documented above rather than hidden.
