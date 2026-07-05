#!/usr/bin/env python3
"""Import MTGJSON AtomicCards into a local SQLite database."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "AtomicCards.json"
DEFAULT_DB = ROOT / "data" / "magic.sqlite"


CARD_INDEXES = [
    "CREATE INDEX {if_not_exists}idx_cards_atomic_name ON cards(atomic_name)",
    "CREATE INDEX {if_not_exists}idx_cards_name ON cards(name)",
    "CREATE INDEX {if_not_exists}idx_cards_name_nocase ON cards(name COLLATE NOCASE)",
    "CREATE INDEX {if_not_exists}idx_cards_ascii_name_nocase ON cards(ascii_name COLLATE NOCASE)",
    "CREATE INDEX {if_not_exists}idx_cards_scryfall_oracle_id ON cards(scryfall_oracle_id)",
    "CREATE INDEX {if_not_exists}idx_cards_scryfall_id ON cards(scryfall_id)",
    "CREATE INDEX {if_not_exists}idx_cards_multiverse_id ON cards(multiverse_id)",
    "CREATE INDEX {if_not_exists}idx_cards_layout ON cards(layout)",
    "CREATE INDEX {if_not_exists}idx_cards_layout_name ON cards(layout, name)",
    "CREATE INDEX {if_not_exists}idx_cards_mana_value ON cards(mana_value)",
    "CREATE INDEX {if_not_exists}idx_cards_mana_value_name ON cards(mana_value, name)",
    "CREATE INDEX {if_not_exists}idx_card_color_identity_color_card ON card_color_identity(color, card_id)",
    "CREATE INDEX {if_not_exists}idx_card_color_identity_card_color ON card_color_identity(card_id, color)",
    "CREATE INDEX {if_not_exists}idx_card_colors_color_card ON card_colors(color, card_id)",
    "CREATE INDEX {if_not_exists}idx_card_colors_card_color ON card_colors(card_id, color)",
    "CREATE INDEX {if_not_exists}idx_card_supertypes_supertype_card ON card_supertypes(supertype, card_id)",
    "CREATE INDEX {if_not_exists}idx_card_supertypes_card_supertype ON card_supertypes(card_id, supertype)",
    "CREATE INDEX {if_not_exists}idx_card_types_type_card ON card_types(type, card_id)",
    "CREATE INDEX {if_not_exists}idx_card_types_card_type ON card_types(card_id, type)",
    "CREATE INDEX {if_not_exists}idx_card_subtypes_subtype_card ON card_subtypes(subtype, card_id)",
    "CREATE INDEX {if_not_exists}idx_card_subtypes_card_subtype ON card_subtypes(card_id, subtype)",
    "CREATE INDEX {if_not_exists}idx_card_keywords_keyword_card ON card_keywords(keyword, card_id)",
    "CREATE INDEX {if_not_exists}idx_card_keywords_card_keyword ON card_keywords(card_id, keyword)",
    "CREATE INDEX {if_not_exists}idx_card_legalities_format_status_card ON card_legalities(format, status, card_id)",
    "CREATE INDEX {if_not_exists}idx_card_legalities_card_format_status ON card_legalities(card_id, format, status)",
    "CREATE INDEX {if_not_exists}idx_card_printings_set_card ON card_printings(set_code, card_id)",
    "CREATE INDEX {if_not_exists}idx_card_printings_card_set ON card_printings(card_id, set_code)",
]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class AtomicCardsReader:
    """Streaming reader for MTGJSON's {"meta": ..., "data": {...}} shape."""

    def __init__(self, path: Path, chunk_size: int = 1024 * 1024) -> None:
        self.path = path
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.file = path.open("r", encoding="utf-8-sig")
        self.buffer = ""
        self.pos = 0
        self.eof = False

    def close(self) -> None:
        self.file.close()

    def read_more(self) -> bool:
        chunk = self.file.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def compact_buffer(self) -> None:
        if self.pos > self.chunk_size:
            self.buffer = self.buffer[self.pos :]
            self.pos = 0

    def ensure(self) -> None:
        while self.pos >= len(self.buffer) and not self.eof:
            self.read_more()

    def skip_ws(self) -> None:
        while True:
            self.ensure()
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer) or self.eof:
                return

    def expect(self, char: str) -> None:
        self.skip_ws()
        self.ensure()
        if self.pos >= len(self.buffer):
            raise ValueError(f"Expected {char!r} at end of file")
        actual = self.buffer[self.pos]
        if actual != char:
            raise ValueError(f"Expected {char!r}, found {actual!r} near offset {self.pos}")
        self.pos += 1
        self.compact_buffer()

    def parse_value(self) -> Any:
        while True:
            self.skip_ws()
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.read_more()
                continue
            self.pos = end
            self.compact_buffer()
            return value

    def parse_string(self) -> str:
        value = self.parse_value()
        if not isinstance(value, str):
            raise ValueError(f"Expected string key, found {type(value).__name__}")
        return value

    def consume_member_separator(self) -> str:
        self.skip_ws()
        self.ensure()
        if self.pos >= len(self.buffer):
            raise ValueError("Expected ',' or '}' at end of file")
        char = self.buffer[self.pos]
        if char not in ",}":
            raise ValueError(f"Expected ',' or '}}', found {char!r}")
        self.pos += 1
        self.compact_buffer()
        return char

    def read(self) -> tuple[dict[str, Any], Iterator[tuple[str, list[dict[str, Any]]]]]:
        self.expect("{")

        meta: dict[str, Any] = {}
        found_data = False

        while True:
            key = self.parse_string()
            self.expect(":")

            if key == "meta":
                meta_value = self.parse_value()
                if not isinstance(meta_value, dict):
                    raise ValueError("Expected top-level meta to be an object")
                meta = meta_value
            elif key == "data":
                found_data = True
                self.expect("{")
                break
            else:
                self.parse_value()

            separator = self.consume_member_separator()
            if separator == "}":
                break

        if not found_data:
            raise ValueError("AtomicCards JSON did not contain a top-level data object")

        return meta, self.iter_cards()

    def iter_cards(self) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        first = True
        while True:
            self.skip_ws()
            self.ensure()
            if self.pos < len(self.buffer) and self.buffer[self.pos] == "}":
                self.pos += 1
                self.skip_ws()
                if self.pos < len(self.buffer) and self.buffer[self.pos] == "}":
                    self.pos += 1
                return

            if not first:
                self.expect(",")
            first = False

            name = self.parse_string()
            self.expect(":")
            cards = self.parse_value()
            if not isinstance(cards, list):
                raise ValueError(f"Expected card list for {name!r}")
            yield name, cards


def create_schema(conn: sqlite3.Connection) -> bool:
    conn.executescript(
        """
        DROP TABLE IF EXISTS card_color_identity;
        DROP TABLE IF EXISTS card_colors;
        DROP TABLE IF EXISTS card_keywords;
        DROP TABLE IF EXISTS card_legalities;
        DROP TABLE IF EXISTS card_printings;
        DROP TABLE IF EXISTS card_subtypes;
        DROP TABLE IF EXISTS card_supertypes;
        DROP TABLE IF EXISTS card_types;
        DROP TABLE IF EXISTS cards;
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS import_stats;

        CREATE TABLE metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE import_stats (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE cards (
          id INTEGER PRIMARY KEY,
          atomic_name TEXT NOT NULL,
          atomic_index INTEGER NOT NULL,
          name TEXT NOT NULL,
          ascii_name TEXT,
          face_name TEXT,
          layout TEXT,
          side TEXT,
          mana_cost TEXT,
          mana_value REAL,
          converted_mana_cost REAL,
          type TEXT,
          oracle_text TEXT,
          power TEXT,
          toughness TEXT,
          loyalty TEXT,
          defense TEXT,
          color_identity_json TEXT NOT NULL,
          colors_json TEXT NOT NULL,
          supertypes_json TEXT NOT NULL,
          types_json TEXT NOT NULL,
          subtypes_json TEXT NOT NULL,
          keywords_json TEXT NOT NULL,
          legalities_json TEXT NOT NULL,
          printings_json TEXT NOT NULL,
          identifiers_json TEXT NOT NULL,
          purchase_urls_json TEXT NOT NULL,
          raw_json TEXT NOT NULL,
          scryfall_oracle_id TEXT,
          scryfall_id TEXT,
          multiverse_id TEXT,
          is_funny INTEGER NOT NULL DEFAULT 0,
          is_reserved INTEGER NOT NULL DEFAULT 0,
          UNIQUE (atomic_name, atomic_index)
        );

        CREATE TABLE card_color_identity (
          card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          color TEXT NOT NULL
        );

        CREATE TABLE card_colors (
          card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          color TEXT NOT NULL
        );

        CREATE TABLE card_supertypes (
          card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          supertype TEXT NOT NULL
        );

        CREATE TABLE card_types (
          card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          type TEXT NOT NULL
        );

        CREATE TABLE card_subtypes (
          card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          subtype TEXT NOT NULL
        );

        CREATE TABLE card_keywords (
          card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          keyword TEXT NOT NULL
        );

        CREATE TABLE card_legalities (
          card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          format TEXT NOT NULL,
          status TEXT NOT NULL
        );

        CREATE TABLE card_printings (
          card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
          set_code TEXT NOT NULL
        );
        """
    )

    fts_enabled = True
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE cards_fts USING fts5(
              name,
              type,
              oracle_text,
              content='cards',
              content_rowid='id'
            )
            """
        )
    except sqlite3.OperationalError:
        fts_enabled = False

    return fts_enabled


def create_indexes(conn: sqlite3.Connection, fts_enabled: bool) -> None:
    for sql in CARD_INDEXES:
        conn.execute(sql.format(if_not_exists=""))
    if fts_enabled:
        conn.execute(
            """
            INSERT INTO cards_fts(rowid, name, type, oracle_text)
            SELECT id, name, type, oracle_text FROM cards
            """
        )


def list_value(card: dict[str, Any], key: str) -> list[str]:
    value = card.get(key) or []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def insert_list(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    card_id: int,
    values: list[str],
) -> None:
    if not values:
        return
    conn.executemany(
        f"INSERT INTO {table} (card_id, {column}) VALUES (?, ?)",
        [(card_id, value) for value in values],
    )


def import_cards(source: Path, database: Path) -> dict[str, Any]:
    start = time.monotonic()
    tmp = database.with_suffix(database.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    reader = AtomicCardsReader(source)
    try:
        meta, card_entries = reader.read()
        database.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(tmp)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = OFF")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA temp_store = MEMORY")
            fts_enabled = create_schema(conn)

            conn.execute("BEGIN")
            for key, value in sorted(meta.items()):
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    (key, compact_json(value) if isinstance(value, (dict, list)) else str(value)),
                )

            card_rows = 0
            atomic_names = 0

            for atomic_name, cards in card_entries:
                atomic_names += 1
                for atomic_index, card in enumerate(cards):
                    if not isinstance(card, dict):
                        continue

                    identifiers = card.get("identifiers") or {}
                    purchase_urls = card.get("purchaseUrls") or {}
                    legalities = card.get("legalities") or {}

                    if not isinstance(identifiers, dict):
                        identifiers = {}
                    if not isinstance(purchase_urls, dict):
                        purchase_urls = {}
                    if not isinstance(legalities, dict):
                        legalities = {}

                    row = (
                        atomic_name,
                        atomic_index,
                        card.get("name") or atomic_name,
                        card.get("asciiName"),
                        card.get("faceName"),
                        card.get("layout"),
                        card.get("side"),
                        card.get("manaCost"),
                        card.get("manaValue"),
                        card.get("convertedManaCost"),
                        card.get("type"),
                        card.get("text"),
                        card.get("power"),
                        card.get("toughness"),
                        card.get("loyalty"),
                        card.get("defense"),
                        compact_json(list_value(card, "colorIdentity")),
                        compact_json(list_value(card, "colors")),
                        compact_json(list_value(card, "supertypes")),
                        compact_json(list_value(card, "types")),
                        compact_json(list_value(card, "subtypes")),
                        compact_json(list_value(card, "keywords")),
                        compact_json(legalities),
                        compact_json(list_value(card, "printings")),
                        compact_json(identifiers),
                        compact_json(purchase_urls),
                        compact_json(card),
                        identifiers.get("scryfallOracleId"),
                        identifiers.get("scryfallId"),
                        identifiers.get("multiverseId"),
                        int(bool(card.get("isFunny"))),
                        int(bool(card.get("isReserved"))),
                    )

                    cursor = conn.execute(
                        """
                        INSERT INTO cards (
                          atomic_name, atomic_index, name, ascii_name, face_name, layout, side,
                          mana_cost, mana_value, converted_mana_cost, type, oracle_text,
                          power, toughness, loyalty, defense,
                          color_identity_json, colors_json, supertypes_json, types_json,
                          subtypes_json, keywords_json, legalities_json, printings_json,
                          identifiers_json, purchase_urls_json, raw_json,
                          scryfall_oracle_id, scryfall_id, multiverse_id,
                          is_funny, is_reserved
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row,
                    )
                    card_id = cursor.lastrowid

                    insert_list(conn, "card_color_identity", "color", card_id, list_value(card, "colorIdentity"))
                    insert_list(conn, "card_colors", "color", card_id, list_value(card, "colors"))
                    insert_list(conn, "card_supertypes", "supertype", card_id, list_value(card, "supertypes"))
                    insert_list(conn, "card_types", "type", card_id, list_value(card, "types"))
                    insert_list(conn, "card_subtypes", "subtype", card_id, list_value(card, "subtypes"))
                    insert_list(conn, "card_keywords", "keyword", card_id, list_value(card, "keywords"))
                    insert_list(conn, "card_printings", "set_code", card_id, list_value(card, "printings"))

                    if legalities:
                        conn.executemany(
                            "INSERT INTO card_legalities (card_id, format, status) VALUES (?, ?, ?)",
                            [(card_id, fmt, str(status)) for fmt, status in sorted(legalities.items())],
                        )

                    card_rows += 1

            create_indexes(conn, fts_enabled)

            stats = {
                "source": str(source),
                "atomic_names": atomic_names,
                "card_rows": card_rows,
                "fts5": int(fts_enabled),
                "import_seconds": round(time.monotonic() - start, 3),
            }
            for key, value in stats.items():
                conn.execute("INSERT INTO import_stats (key, value) VALUES (?, ?)", (key, str(value)))

            conn.commit()
            conn.execute("VACUUM")
        finally:
            conn.close()
    finally:
        reader.close()

    os.replace(tmp, database)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help=f"AtomicCards JSON path. Default: {DEFAULT_SOURCE}")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help=f"SQLite output path. Default: {DEFAULT_DB}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    database = args.database.resolve()

    if not source.exists():
        raise SystemExit(f"Missing source file: {source}")

    stats = import_cards(source, database)
    print(f"Wrote {database}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
