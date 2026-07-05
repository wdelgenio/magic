#!/usr/bin/env python3
"""Build and search local card-text vectors stored in SQLite.

The default embedding model is a deterministic local hashing model. It is
dependency-free and useful for validating the SQLite vector plumbing locally.
The schema stores model names explicitly so higher-quality embeddings can be
added later without changing callers.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "magic.sqlite"
DEFAULT_MODEL = "local-hash-v1"
DEFAULT_DIMENSIONS = 384
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'+/-]*")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS card_semantic_documents (
  card_id INTEGER PRIMARY KEY REFERENCES cards(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS card_embeddings (
  card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  embedding BLOB NOT NULL,
  text_hash TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (card_id, model)
);

CREATE INDEX IF NOT EXISTS idx_card_embeddings_model
  ON card_embeddings(model);

CREATE INDEX IF NOT EXISTS idx_card_embeddings_model_text_hash
  ON card_embeddings(model, text_hash);
"""


def connect(database: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(database)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def card_document(row: sqlite3.Row) -> str:
    parts = [
        row["name"],
        row["mana_cost"],
        row["type"],
        row["oracle_text"],
        row["keywords_json"],
        row["color_identity_json"],
    ]
    return "\n".join(part for part in parts if part)


def init_documents(conn: sqlite3.Connection) -> int:
    conn.executescript(SCHEMA_SQL)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, name, mana_cost, type, oracle_text, keywords_json, color_identity_json
        FROM cards
        """
    )

    count = 0
    conn.execute("BEGIN")
    for row in rows:
        document = card_document(row)
        conn.execute(
            """
            INSERT INTO card_semantic_documents(card_id, text, text_hash, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(card_id) DO UPDATE SET
              text = excluded.text,
              text_hash = excluded.text_hash,
              updated_at = CURRENT_TIMESTAMP
            """,
            (row["id"], document, text_hash(document)),
        )
        count += 1
    conn.commit()
    return count


def tokens(text: str) -> list[str]:
    normalized = text.lower().replace("−", "-")
    words = TOKEN_RE.findall(normalized)
    features = list(words)
    features.extend(f"{left} {right}" for left, right in zip(words, words[1:]))
    return features


def hashed_embedding(text: str, dimensions: int) -> array.array:
    vector = [0.0] * dimensions
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign

    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude:
        vector = [value / magnitude for value in vector]
    return array.array("f", vector)


def vector_blob(vector: array.array) -> bytes:
    return vector.tobytes()


def vector_from_blob(blob: bytes) -> memoryview:
    return memoryview(blob).cast("f")


def dot_score(query: array.array, blob: bytes) -> float:
    candidate = vector_from_blob(blob)
    if len(candidate) != len(query):
        return -1.0
    return float(sum(left * right for left, right in zip(query, candidate)))


def build_embeddings(conn: sqlite3.Connection, model: str, dimensions: int, rebuild: bool) -> int:
    conn.executescript(SCHEMA_SQL)
    if rebuild:
        conn.execute("DELETE FROM card_embeddings WHERE model = ?", (model,))
        conn.commit()

    rows = conn.execute(
        """
        SELECT d.card_id, d.text, d.text_hash
        FROM card_semantic_documents d
        LEFT JOIN card_embeddings e
          ON e.card_id = d.card_id
         AND e.model = ?
         AND e.text_hash = d.text_hash
         AND e.dimensions = ?
        WHERE e.card_id IS NULL
        """,
        (model, dimensions),
    )

    count = 0
    conn.execute("BEGIN")
    for card_id, text, current_hash in rows:
        embedding = hashed_embedding(text, dimensions)
        conn.execute(
            """
            INSERT INTO card_embeddings(card_id, model, dimensions, embedding, text_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(card_id, model) DO UPDATE SET
              dimensions = excluded.dimensions,
              embedding = excluded.embedding,
              text_hash = excluded.text_hash,
              updated_at = CURRENT_TIMESTAMP
            """,
            (card_id, model, dimensions, vector_blob(embedding), current_hash),
        )
        count += 1
        if count % 1000 == 0:
            conn.commit()
            conn.execute("BEGIN")
    conn.commit()
    return count


def ensure_model(conn: sqlite3.Connection, model: str) -> int:
    row = conn.execute(
        """
        SELECT dimensions
        FROM card_embeddings
        WHERE model = ?
        LIMIT 1
        """,
        (model,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"No embeddings found for model {model!r}. Run: python scripts/semantic_cards.py build")
    return int(row[0])


def search(conn: sqlite3.Connection, query: str, model: str, limit: int) -> list[sqlite3.Row]:
    dimensions = ensure_model(conn, model)
    query_vector = hashed_embedding(query, dimensions)
    conn.create_function("semantic_score", 1, lambda blob: dot_score(query_vector, blob))
    conn.row_factory = sqlite3.Row
    return list(
        conn.execute(
            """
            SELECT
              c.id,
              c.name,
              c.mana_cost,
              c.type,
              c.oracle_text,
              semantic_score(e.embedding) AS score
            FROM card_embeddings e
            JOIN cards c ON c.id = e.card_id
            WHERE e.model = ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (model, limit),
        )
    )


def print_results(rows: Iterable[sqlite3.Row]) -> None:
    for row in rows:
        oracle = (row["oracle_text"] or "").replace("\n", " ")
        if len(oracle) > 140:
            oracle = oracle[:137] + "..."
        mana = f" {row['mana_cost']}" if row["mana_cost"] else ""
        print(f"{row['score']:.4f}\t{row['name']}{mana}\t{row['type']}")
        if oracle:
            print(f"    {oracle}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help=f"SQLite database path. Default: {DEFAULT_DB}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create semantic tables and refresh card text documents.")

    build = subparsers.add_parser("build", help="Build local card-text vector embeddings.")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    build.add_argument("--rebuild", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search cards by semantic-ish text similarity.")
    search_parser.add_argument("query", nargs="+")
    search_parser.add_argument("--model", default=DEFAULT_MODEL)
    search_parser.add_argument("--limit", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = args.database.resolve()
    if not database.exists():
        raise SystemExit(f"Missing database: {database}")

    conn = connect(database)
    try:
        if args.command == "init":
            count = init_documents(conn)
            print(f"Refreshed {count} card semantic documents")
        elif args.command == "build":
            init_documents(conn)
            count = build_embeddings(conn, args.model, args.dimensions, args.rebuild)
            conn.execute("ANALYZE")
            print(f"Built {count} embeddings for model {args.model!r}")
        elif args.command == "search":
            query = " ".join(args.query)
            print_results(search(conn, query, args.model, args.limit))
        else:
            raise SystemExit(f"Unknown command: {args.command}")
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
