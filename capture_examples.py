"""
capture_examples.py — runs the two example calls required by the task
(one that should trigger retrieval, one that should not) against the FastAPI
app with MOCK_LLM left at its default, and writes the raw JSON responses to
example_calls.md for the README.

Usage (normal — uses the real all-MiniLM-L6-v2 embedder + ChromaDB):
    python capture_examples.py

Usage (offline demo — no model download, for environments without internet
access to huggingface.co; output is clearly labeled as illustrative):
    python capture_examples.py --fake-embedder
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-embedder", action="store_true",
                         help="Use the deterministic test embedder instead of "
                              "downloading all-MiniLM-L6-v2 (offline demo only).")
    args = parser.parse_args()

    os.environ.setdefault("MOCK_LLM", "1")

    import vectorstore
    if args.fake_embedder:
        from tests.fake_embedder import fake_embed_fn
        vectorstore.default_embed_fn = fake_embed_fn
        note = ("> **Note:** captured with the lightweight deterministic test "
                "embedder (`--fake-embedder`), not the real all-MiniLM-L6-v2 "
                "model, because this environment has no internet access to "
                "huggingface.co. Re-run `python capture_examples.py` (no flag) "
                "on a machine with normal internet access before final "
                "submission to regenerate this file with the real model.\n\n")
    else:
        note = ""

    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)

    examples = [
        ("Policy question (should trigger retrieval)", "How much does standard delivery cost?"),
        ("General question (should NOT trigger retrieval)", "What's a good recipe for pancakes?"),
    ]

    lines = ["# Example /ask calls\n", note]
    for label, query in examples:
        resp = client.post("/ask", json={"query": query})
        lines.append(f"## {label}\n")
        lines.append(f"**Request:**\n```json\n{json.dumps({'query': query}, indent=2)}\n```\n")
        lines.append(f"**Response ({resp.status_code}):**\n```json\n"
                      f"{json.dumps(resp.json(), indent=2)}\n```\n")

    out_path = HERE / "example_calls.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
