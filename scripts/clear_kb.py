#!/usr/bin/env python
"""Clear vector store collections.

Usage:
    python -m scripts.clear_kb --general
    python -m scripts.clear_kb --personal --user-id default_user
    python -m scripts.clear_kb --all
"""

from __future__ import annotations

import argparse

from app.services.vectorstore import VectorStoreService


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear vector store collections.")
    parser.add_argument("--general", action="store_true", help="Clear the general KB.")
    parser.add_argument("--personal", action="store_true", help="Clear a user's personal KB.")
    parser.add_argument("--user-id", type=str, default=None, help="User ID (required with --personal).")
    parser.add_argument("--all", action="store_true", help="Clear all collections.")
    args = parser.parse_args()

    if not (args.general or args.personal or args.all):
        parser.error("Specify --general, --personal, or --all.")

    if args.personal and not args.user_id:
        parser.error("--user-id is required with --personal.")

    store = VectorStoreService()

    if args.all:
        confirm = input("This will delete ALL collections (general + all personal). Continue? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

        cleared = store.clear_collection("general")
        print(f"Cleared general KB ({cleared} chunks)")

        user_ids = store.list_all_personal_collections()
        for uid in user_ids:
            cleared = store.clear_collection("personal", user_id=uid)
            print(f"Cleared personal KB for '{uid}' ({cleared} chunks)")

        if not user_ids:
            print("No personal collections found.")
        print("Done.")
        return

    if args.general:
        confirm = input("Clear the general KB? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return
        cleared = store.clear_collection("general")
        print(f"Cleared general KB ({cleared} chunks)")

    if args.personal:
        confirm = input(f"Clear personal KB for '{args.user_id}'? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return
        cleared = store.clear_collection("personal", user_id=args.user_id)
        print(f"Cleared personal KB for '{args.user_id}' ({cleared} chunks)")


if __name__ == "__main__":
    main()
