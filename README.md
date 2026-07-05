# Magic

Personal Magic: The Gathering deck-building repository.

## Repository layout

### `rules/mtg-rules/`

Git submodule for [`chaoticgoodcomputing/mtg-rules`](https://github.com/chaoticgoodcomputing/mtg-rules), an unofficial automated Markdown mirror of the official Magic: The Gathering Comprehensive Rules.

### `proxy-orders/`

Reusable text files for proxy planning and MPCFill-style imports.

Canonical files:

- `staples.txt`
  - Generic Commander staples and reusable cards worth owning as proxies across multiple decks.

- `lands.txt`
  - High-value mana base pieces, utility lands, and bulk basics.

- `commanders.txt`
  - Commanders and commander-adjacent legendary creatures.

- `quantities.txt`
  - Notes for cards where multiple proxy copies are useful.

## Buy-list files

Deck-specific `*-buy-list.txt` files are temporary export files, not canonical sources.

When placing a proxy order, generate MPCFill or other buy-list exports from the canonical files above, use them for the order, and delete them afterward. This avoids maintaining duplicate lists in multiple formats.

## Card database

Place MTGJSON `AtomicCards.json` at `data/AtomicCards.json`, then build a local SQLite database:

```sh
python scripts/import_atomic_cards.py
```

The importer writes `data/magic.sqlite`. The MTGJSON export, generated SQLite
database, and local GGUF model are tracked with Git LFS; SQLite runtime sidecars
are still ignored.

To refresh indexes on an existing database without rebuilding:

```sh
python scripts/add_card_indexes.py
```

To build and search local card-text vectors:

```sh
python scripts/semantic_cards.py build
python scripts/semantic_cards.py search "double strike combat damage"
```

For better local semantic search, install `llama-cpp`, put the Nomic GGUF
embedding model at
`data/models/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf`,
then build Nomic embeddings:

```sh
python scripts/nomic_cards.py build
python scripts/nomic_cards.py search "return creature from graveyard to battlefield"
```

The Nomic build stores 768-dimensional vectors in the same `card_embeddings`
table under model key `nomic-embed-text-v1.5.Q4_K_M`. It is fully local, but
slow on a phone; run the same command on a desktop if building all cards takes
too long.

## Future ideas

- Decklists
- Upgrade paths
- Synergy notes
- Scryfall links
- Playtest notes
- MPCFill exports
