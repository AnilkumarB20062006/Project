# Example /ask calls

> **Note:** captured with the lightweight deterministic test embedder (`--fake-embedder`), not the real all-MiniLM-L6-v2 model, because this environment has no internet access to huggingface.co. Re-run `python capture_examples.py` (no flag) on a machine with normal internet access before final submission to regenerate this file with the real model.


## Policy question (should trigger retrieval)

**Request:**
```json
{
  "query": "How much does standard delivery cost?"
}
```

**Response (200):**
```json
{
  "answer": "Based on the retrieved context: Delivery Policy: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vol",
  "sources": [
    "doc_01",
    "doc_03",
    "doc_04"
  ],
  "confidence": 1.0
}
```

## General question (should NOT trigger retrieval)

**Request:**
```json
{
  "query": "What's a good recipe for pancakes?"
}
```

**Response (200):**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
