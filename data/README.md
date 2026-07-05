# Local Data

This directory is for generated or downloaded card data used by local tools.

Git LFS-managed files:

- `AtomicCards.json` - MTGJSON AtomicCards bulk export.
- `*.sqlite` - generated SQLite databases.
- `models/**/*.gguf` - local embedding model files.

Ignored local files:

- `*.sqlite-*` - SQLite runtime sidecars.

Build the local card database with:

```sh
python scripts/import_atomic_cards.py
```

Refresh indexes on an existing database without rebuilding it:

```sh
python scripts/add_card_indexes.py
```

Build local card-text vectors for SQLite-backed similarity search:

```sh
python scripts/semantic_cards.py build
```

Search those vectors:

```sh
python scripts/semantic_cards.py search "return creature from graveyard to battlefield"
```

The default model, `local-hash-v1`, is a dependency-free local vector model
intended to keep the database/search pipeline portable. Higher-quality
embeddings can be added as another model in the same `card_embeddings` table.

Build higher-quality local Nomic embeddings with llama.cpp:

```sh
python scripts/nomic_cards.py build
```

Search Nomic embeddings:

```sh
python scripts/nomic_cards.py search "return creature from graveyard to battlefield"
```

Expected model file:

```text
data/models/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf
```

The Nomic helper starts a local `llama-server` by default so the model stays
loaded during a build. Use `--server-url http://127.0.0.1:8091` to point at an
already-running server, or `--backend cli` for the slower one-shot
`llama-embedding` fallback.
