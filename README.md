# Magic

Personal Magic: The Gathering deck-building repository.

## Repository layout

### `proxy-orders/`

Reusable text files for proxy planning and MPCFill-style imports.

Canonical files:

- `mpcfill-master-buy-list.txt`
  - Large combined proxy candidate list across decks.
  - Intended to be easy to paste into MPCFill or another proxy-printer import flow.

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

When placing a proxy order, generate buy lists from the canonical files above, use them for the order, and delete them afterward. This avoids maintaining duplicate lists in multiple formats.

## Future ideas

- Decklists
- Upgrade paths
- Synergy notes
- Scryfall links
- Playtest notes
- MPCFill exports
