#!/usr/bin/env python3
"""Add or refresh practical indexes on the local Magic card SQLite database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from import_atomic_cards import CARD_INDEXES, DEFAULT_DB


REDUNDANT_LEGACY_INDEXES = [
    "idx_card_color_identity_color",
    "idx_card_colors_color",
    "idx_card_types_type",
    "idx_card_subtypes_subtype",
    "idx_card_keywords_keyword",
    "idx_card_legalities_format_status",
    "idx_card_printings_set_code",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help=f"SQLite database path. Default: {DEFAULT_DB}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = args.database.resolve()
    if not database.exists():
        raise SystemExit(f"Missing database: {database}")

    conn = sqlite3.connect(database)
    try:
        for index_name in REDUNDANT_LEGACY_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        for sql in CARD_INDEXES:
            conn.execute(sql.format(if_not_exists="IF NOT EXISTS "))
        conn.commit()
        conn.execute("ANALYZE")
        conn.execute("PRAGMA optimize")
    finally:
        conn.close()

    print(f"Indexed {database}")


if __name__ == "__main__":
    main()
