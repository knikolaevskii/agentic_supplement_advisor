# Seed Documents

Place `.txt` and `.pdf` files in the appropriate subfolder:

- **`general/`** — shared knowledge base (research articles, supplement guides, general health info)
- **`personal/`** — personal documents (lab results, health records, supplement plans)

Run the seed script to ingest them:

```bash
python -m scripts.seed_kb                                # ingest both folders
python -m scripts.seed_kb --general-only                  # general folder only
python -m scripts.seed_kb --personal-only --user-id alice # personal folder for user "alice"
python -m scripts.seed_kb --clear                         # wipe collections + re-ingest
python -m scripts.seed_kb --force                         # re-ingest all without wiping
```

Files already ingested are skipped on re-run (matched by filename hash).
