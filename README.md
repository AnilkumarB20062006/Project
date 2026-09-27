# Module 3 — Support Assistant (`/support_assistant`)

A small RAG service that answers customer questions about Zepto's own
policies, grounded in an 8-document policy corpus, orchestrated with
LangGraph, and served over FastAPI.

## Files

| File | Purpose |
|---|---|
| `docs/doc_01.txt` … `doc_08.txt` | The required policy corpus (delivery, returns, membership, tracking, cancellation, damaged items, gift cards, support hours). |
| `vectorstore.py` | Chunking (one chunk per doc — all comfortably under the fixed-size fallback threshold), embedding with `sentence-transformers/all-MiniLM-L6-v2`, and a ChromaDB-backed `VectorStore` with cosine-similarity top-k retrieval. |
| `prompt_template.py` | The structured role → context → task → format → length prompt, with a negative constraint and a few-shot example, used by the optional real-LLM extension. |
| `schema.py` | Pydantic `AskRequest` / `AskResponse` models and the `GraphState` TypedDict. |
| `graph.py` | The LangGraph `StateGraph`: `classify_intent` → (conditional edge) → `retrieve_and_answer` **or** `direct_answer`. Every generation step branches on the `MOCK_LLM` env var. |
| `main.py` | FastAPI app exposing `POST /ask`. |
| `capture_examples.py` | Captures the 2 required example calls into `example_calls.md`. |
| `Dockerfile` | Builds + serves the FastAPI app locally on port 7860. |
| `tests/` | Offline pytest suite (chunking, retrieval, graph routing, FastAPI endpoint) using a deterministic fake embedder — no model download or network access required to run it. |

## Run it

```bash
pip install -r requirements.txt         # or the repo's consolidated requirements.txt

# 1. Ingest the corpus into ChromaDB (downloads all-MiniLM-L6-v2 on first run)
python vectorstore.py

# 2. Serve the API (MOCK_LLM defaults to 1 — the graded baseline, no LLM/API key needed)
uvicorn main:app --host 0.0.0.0 --port 7860

# 3. Call it
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "How much does standard delivery cost?"}'

# Regenerate the recorded example calls for the README:
python capture_examples.py
```

Docker:

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

Offline test suite (no model download needed — uses a deterministic fake embedder):

```bash
pytest tests/ -v
```

## Architecture — the RAG pipeline, stage by stage

```
 ingestion            embedding                 retrieval                 generation
┌──────────┐        ┌──────────────┐        ┌──────────────────┐     ┌────────────────────┐
│ docs/*.txt│──────▶│ all-MiniLM-L6│──────▶│ ChromaDB          │────▶│ classify_intent     │
│ (8 files) │  chunk │ -v2 (local)  │ embed  │ collection        │     │ (LangGraph node)    │
└──────────┘        └──────────────┘        │ 'zepto_policies'  │     └─────────┬───────────┘
                                             └──────────────────┘               │
                                                      ▲                conditional edge
                                                      │                         │
                                              query embedding          ┌────────┴─────────┐
                                              (same model)             ▼                  ▼
                                                      │        retrieve_and_answer   direct_answer
                                                      │        (top-3 cosine lookup) (no retrieval)
                                                      └────────────────┘                  │
                                                               │                          │
                                                               ▼                          ▼
                                                        MOCK_LLM branch            MOCK_LLM branch
                                                     (templated answer OR          (canned string OR
                                                      real-LLM w/ prompt_template)  real-LLM direct)
                                                               │                          │
                                                               └──────────┬───────────────┘
                                                                          ▼
                                                                  Pydantic AskResponse
                                                                  {answer, sources, confidence}
                                                                          ▼
                                                                   FastAPI POST /ask
```

1. **Ingestion** — `vectorstore.load_and_chunk_documents()` reads all 8 `docs/*.txt`
   files. Given their short length, each document becomes exactly one chunk
   (id = filename stem, e.g. `doc_01`); a fixed-size fallback split exists
   for any document that would exceed the length threshold.
2. **Embedding** — `vectorstore.default_embed_fn()` loads
   `sentence-transformers/all-MiniLM-L6-v2` once (module-level lazy
   singleton) and encodes chunk text into normalized vectors, entirely
   on-machine, no API key.
3. **Storage** — `VectorStore` wraps a ChromaDB `PersistentClient` /
   collection named `zepto_policies` (`vectorstore.py`, persisted under
   `chroma_store/`).
4. **Retrieval** — `VectorStore.query()` embeds the incoming query with the
   *same* model and asks ChromaDB for the top-3 nearest chunks by cosine
   similarity. This step runs identically in both `MOCK_LLM` states — it
   needs no API key and no network call once the model is cached locally.
5. **Routing** — the `classify_intent` LangGraph node (keyword heuristic in
   mock mode) decides `policy_question` vs `general_question`; a conditional
   edge sends the state to `retrieve_and_answer` or `direct_answer`
   accordingly. This routing logic is identical regardless of `MOCK_LLM`.
6. **Generation** — this is the *only* stage that branches on `MOCK_LLM`:
   - **`MOCK_LLM` unset / `"1"` (default, graded baseline):**
     `retrieve_and_answer` returns a canned
     `f"Based on the retrieved context: {top_chunk_snippet}"` string built
     from the top retrieved chunk; `direct_answer` returns a fixed
     `"I can only answer questions about Zepto policies right now."` — no
     LLM call, no network call, in either node.
   - **`MOCK_LLM=0` (optional, ungraded extension):** the same nodes instead
     build the structured prompt (`prompt_template.build_prompt`, using the
     retrieved chunks as context) and call a real LLM; the raw output is
     validated against `AskResponse` and retried up to 2 more times with a
     corrective instruction on validation failure (integration point:
     `_llm_generate_grounded_answer` / `_llm_direct_answer` /
     `_llm_classify_intent` in `graph.py`, deliberately left as
     `NotImplementedError` since the graded baseline never calls them).
7. **Schema enforcement** — in both modes the final state is packed into a
   Pydantic `AskResponse` (`answer`, `sources`, `confidence`); in mock mode
   this is populated deterministically in code, so there's no LLM output to
   fail validation.
8. **API** — `main.py`'s `POST /ask` accepts an `AskRequest({"query": str})`
   and returns the validated `AskResponse`.

## A note on this sandbox

The sandbox this was built in cannot reach `huggingface.co` (its network
egress is restricted to a short package-registry allow-list), so the real
`all-MiniLM-L6-v2` weights could not be downloaded here. Every other piece
— chunking, ChromaDB storage/retrieval, the LangGraph graph and its
conditional routing, the FastAPI endpoint, and Pydantic validation — **was**
tested end to end here, using a deterministic, dependency-free fake
embedder swapped in only for tests (`tests/fake_embedder.py`, injected via
the `embed_fn` parameter `VectorStore` already exposes for this purpose).
All 10 + 6 tests in `tests/` pass. `example_calls.md` in this folder was
captured the same way and is clearly labeled — run
`python capture_examples.py` (no flag) on a machine with normal internet
access to regenerate it with the real model before submission.
