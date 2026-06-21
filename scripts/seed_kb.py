#!/usr/bin/env python
"""Seed the general knowledge base from pdfs/.

Ingests every PDF in the pdfs/ directory into the general ChromaDB collection
and copies each file to data/documents/ so the UI viewer can display it.

Doc IDs are derived deterministically from the filename, so this script is
safe to re-run — already-ingested files are skipped unless --force is passed.

Usage:
    python -m scripts.seed_kb              # ingest new files only
    python -m scripts.seed_kb --force      # wipe + re-ingest everything
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil

from app.services.document import process_document
from app.services.vectorstore import VectorStoreService

PDFS_DIR = pathlib.Path("pdfs")
DOCS_DIR = pathlib.Path("data/documents")


def _doc_id(filename: str) -> str:
    """Deterministic 12-char doc_id derived from the filename."""
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the general knowledge base.")
    parser.add_argument(
        "--force", action="store_true",
        help="Clear the general collection and re-ingest all files.",
    )
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    store = VectorStoreService()

    if args.force:
        cleared = store.clear_collection("general")
        print(f"Cleared general KB ({cleared} chunks removed)")

    existing_doc_ids: set[str] = {
        doc["doc_id"] for doc in store.list_general_documents()
    }

    pdf_files = sorted(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDFS_DIR}/")
        return

    print(f"Found {len(pdf_files)} PDFs in {PDFS_DIR}/\n")

    total_files = total_chunks = 0
    failed: list[str] = []

    for pdf_path in pdf_files:
        doc_id = _doc_id(pdf_path.name)

        if doc_id in existing_doc_ids:
            print(f"  skip  {pdf_path.name}")
            continue

        try:
            file_bytes = pdf_path.read_bytes()
            chunks = process_document(file_bytes, pdf_path.name, doc_id)
            store.add_documents(chunks, collection_type="general")
            shutil.copy2(pdf_path, DOCS_DIR / f"{doc_id}.pdf")
            print(f"  ok    {pdf_path.name}  ({len(chunks)} chunks, id={doc_id})")
            total_files += 1
            total_chunks += len(chunks)
        except Exception as exc:
            print(f"  FAIL  {pdf_path.name}: {exc}")
            failed.append(pdf_path.name)

    print(f"\nIngested {total_files} file(s), {total_chunks} chunks total.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
