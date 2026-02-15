#!/usr/bin/env python
"""Seed the knowledge base from files in data/seed_documents/.

Reads from two subfolders:
  data/seed_documents/general/   → general KB (shared)
  data/seed_documents/personal/  → personal KB (per user)

Usage:
    python -m scripts.seed_kb                                # ingest both folders
    python -m scripts.seed_kb --clear                        # wipe + re-ingest
    python -m scripts.seed_kb --force                        # re-ingest all
    python -m scripts.seed_kb --general-only                 # general folder only
    python -m scripts.seed_kb --personal-only --user-id alice
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

from app.services.document import chunk_text, extract_text
from app.services.vectorstore import VectorStoreService

SEED_DIR = pathlib.Path("data/seed_documents")
GENERAL_DIR = SEED_DIR / "general"
PERSONAL_DIR = SEED_DIR / "personal"
ALLOWED_EXTENSIONS = {".txt", ".pdf"}


def _doc_id_from_filename(filename: str) -> str:
    """Deterministic doc_id derived from the filename."""
    return hashlib.md5(filename.encode()).hexdigest()[:12]


def _discover_files(folder: pathlib.Path) -> list[pathlib.Path]:
    """Return sorted list of ingestible files in *folder*."""
    if not folder.is_dir():
        return []
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _seed_folder(
    store: VectorStoreService,
    folder: pathlib.Path,
    collection_type: str,
    user_id: str | None,
    existing_doc_ids: set[str],
) -> tuple[int, int]:
    """Ingest files from *folder*. Returns (files_ingested, chunks_ingested)."""
    files = _discover_files(folder)
    if not files:
        print(f"  No .txt or .pdf files found in {folder}/")
        return 0, 0

    total_files = 0
    total_chunks = 0

    for filepath in files:
        doc_id = _doc_id_from_filename(filepath.name)

        if doc_id in existing_doc_ids:
            print(f"  Skipped (already ingested): {filepath.name}")
            continue

        file_bytes = filepath.read_bytes()
        text = extract_text(file_bytes, filepath.name)

        if not text.strip():
            print(f"  Skipped (empty): {filepath.name}")
            continue

        title = filepath.stem.replace("_", " ").replace("-", " ").title()
        pieces = chunk_text(text, chunk_size=500, overlap=50)
        chunks = [
            {
                "text": piece,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "title": title,
                    "source": filepath.name,
                    "page": 1,
                },
            }
            for i, piece in enumerate(pieces)
        ]

        store.add_documents(chunks, collection_type=collection_type, user_id=user_id)
        total_files += 1
        total_chunks += len(chunks)
        print(f"  Ingested [{collection_type}]: {filepath.name} ({len(chunks)} chunks)")

    return total_files, total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the knowledge base.")
    parser.add_argument(
        "--clear", action="store_true",
        help="Wipe collections before seeding.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest all files even if already present.",
    )
    parser.add_argument(
        "--user-id", type=str, default="default_user",
        help="User ID for personal documents (default: default_user).",
    )
    parser.add_argument(
        "--general-only", action="store_true",
        help="Only seed the general folder.",
    )
    parser.add_argument(
        "--personal-only", action="store_true",
        help="Only seed the personal folder.",
    )
    args = parser.parse_args()

    seed_general = not args.personal_only
    seed_personal = not args.general_only

    store = VectorStoreService()

    # Optionally clear collections.
    if args.clear:
        if seed_general:
            cleared = store.clear_collection("general")
            print(f"Cleared general KB ({cleared} chunks removed)")
        if seed_personal:
            cleared = store.clear_collection("personal", user_id=args.user_id)
            print(f"Cleared personal KB for '{args.user_id}' ({cleared} chunks removed)")

    # Seed general folder.
    gen_files = gen_chunks = 0
    if seed_general:
        existing: set[str] = set()
        if not args.clear and not args.force:
            for doc in store.list_general_documents():
                existing.add(doc["doc_id"])
        gen_files, gen_chunks = _seed_folder(store, GENERAL_DIR, "general", None, existing)

    # Seed personal folder.
    per_files = per_chunks = 0
    if seed_personal:
        existing = set()
        if not args.clear and not args.force:
            for doc in store.list_user_documents(args.user_id):
                existing.add(doc["doc_id"])
        per_files, per_chunks = _seed_folder(
            store, PERSONAL_DIR, "personal", args.user_id, existing,
        )

    # Summary.
    parts: list[str] = []
    if seed_general:
        parts.append(f"General: {gen_files} files ({gen_chunks} chunks)")
    if seed_personal:
        parts.append(f"Personal [{args.user_id}]: {per_files} files ({per_chunks} chunks)")
    print(f"\n{', '.join(parts)}.")


if __name__ == "__main__":
    main()
