#!/usr/bin/env python3
"""Build and search Nomic GGUF card embeddings with llama.cpp."""

from __future__ import annotations

import argparse
import array
import contextlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

from semantic_cards import DEFAULT_DB, SCHEMA_SQL, dot_score, init_documents


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "data" / "models" / "nomic-embed-text-v1.5-GGUF" / "nomic-embed-text-v1.5.Q4_K_M.gguf"
DEFAULT_MODEL_NAME = "nomic-embed-text-v1.5.Q4_K_M"
DEFAULT_BATCH_SIZE = 64
DEFAULT_CTX_SIZE = 8192
DEFAULT_LIMIT = 10
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8091
DEFAULT_REQUEST_TIMEOUT = 600
DEFAULT_STARTUP_TIMEOUT = 90


EmbedTexts = Callable[[list[str]], list[list[float]]]


def vector_blob(values: list[float]) -> bytes:
    return array.array("f", values).tobytes()


def one_line(text: str) -> str:
    return " ".join(text.split())


def prefixed_document(text: str) -> str:
    return f"search_document: {one_line(text)}"


def prefixed_query(text: str) -> str:
    return f"search_query: {one_line(text)}"


def parse_raw_embeddings(stdout: str, expected_count: int) -> list[list[float]]:
    embeddings = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        embeddings.append([float(value) for value in line.split()])

    if len(embeddings) != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, got {len(embeddings)}")
    if embeddings:
        dimensions = len(embeddings[0])
        for index, embedding in enumerate(embeddings, start=1):
            if len(embedding) != dimensions:
                raise ValueError(
                    f"Embedding {index} has {len(embedding)} dimensions, expected {dimensions}"
                )
    return embeddings


def parse_embedding_payload(payload: dict, expected_count: int) -> list[list[float]]:
    rows = sorted(payload.get("data", []), key=lambda row: row["index"])
    embeddings = [row["embedding"] for row in rows]
    if len(embeddings) != expected_count:
        raise ValueError(f"Expected {expected_count} embeddings, got {len(embeddings)}")
    if embeddings:
        dimensions = len(embeddings[0])
        for index, embedding in enumerate(embeddings, start=1):
            if len(embedding) != dimensions:
                raise ValueError(
                    f"Embedding {index} has {len(embedding)} dimensions, expected {dimensions}"
                )
    return embeddings


def request_server_embeddings(
    server_url: str,
    texts: list[str],
    model_name: str,
    timeout: float,
) -> list[list[float]]:
    payload = json.dumps({"input": texts, "model": model_name}).encode("utf-8")
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server returned HTTP {exc.code}: {body[:500]}") from exc
    return parse_embedding_payload(response_payload, len(texts))


def run_llama_embeddings(
    model_path: Path,
    texts: list[str],
    ctx_size: int,
    llama_batch_size: int,
) -> list[list[float]]:
    if not texts:
        return []

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        input_path = Path(handle.name)
        for text in texts:
            handle.write(text)
            handle.write("\n")
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        command = [
            "llama-embedding",
            "-m",
            str(model_path),
            "-f",
            str(input_path),
            "--pooling",
            "mean",
            "--embd-output-format",
            "raw",
            "--embd-normalize",
            "2",
            "--no-warmup",
            "--log-verbosity",
            "1",
            "--ctx-size",
            str(ctx_size),
            "--batch-size",
            str(llama_batch_size),
            "--rope-scaling",
            "yarn",
            "--rope-freq-scale",
            ".75",
        ]
        with output_path.open("w", encoding="utf-8") as stdout:
            subprocess.run(command, check=True, stdout=stdout, stderr=subprocess.PIPE, text=True)
        output = output_path.read_text(encoding="utf-8")
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    return parse_raw_embeddings(output, len(texts))


def run_llama_embeddings_with_retry(
    embed_texts: EmbedTexts,
    texts: list[str],
) -> list[list[float]]:
    try:
        return embed_texts(texts)
    except (RuntimeError, TimeoutError, urllib.error.URLError, ValueError) as exc:
        if len(texts) <= 1:
            raise
        midpoint = len(texts) // 2
        print(
            f"Batch of {len(texts)} did not return cleanly ({exc}); retrying as "
            f"{midpoint} + {len(texts) - midpoint}",
            flush=True,
        )
        return (
            run_llama_embeddings_with_retry(embed_texts, texts[:midpoint])
            + run_llama_embeddings_with_retry(embed_texts, texts[midpoint:])
        )


def server_command(
    model_path: Path,
    host: str,
    port: int,
    ctx_size: int,
    llama_batch_size: int,
) -> list[str]:
    return [
        "llama-server",
        "-m",
        str(model_path),
        "--embedding",
        "--pooling",
        "mean",
        "--embd-normalize",
        "2",
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--batch-size",
        str(llama_batch_size),
        "--rope-scaling",
        "yarn",
        "--rope-freq-scale",
        ".75",
        "--no-warmup",
        "--no-ui",
        "--log-verbosity",
        "1",
    ]


@contextlib.contextmanager
def managed_llama_server(
    model_path: Path,
    host: str,
    port: int,
    ctx_size: int,
    llama_batch_size: int,
    model_name: str,
    request_timeout: float,
    startup_timeout: float,
) -> Iterable[str]:
    command = server_command(model_path, host, port, ctx_size, llama_batch_size)
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        server_url = f"http://{host}:{port}"
        deadline = time.monotonic() + startup_timeout
        try:
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    raise SystemExit(f"llama-server exited during startup:\n{log.read()[-2000:]}")
                try:
                    request_server_embeddings(
                        server_url,
                        [prefixed_query("ping")],
                        model_name,
                        timeout=min(5, request_timeout),
                    )
                    break
                except (RuntimeError, urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        log.seek(0)
                        raise SystemExit(f"Timed out waiting for llama-server:\n{log.read()[-2000:]}")
                    time.sleep(0.5)
            yield server_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def connect(database: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(database)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    return conn


def ensure_runtime(model_path: Path) -> None:
    if not model_path.exists():
        raise SystemExit(f"Missing model file: {model_path}")
    if shutil.which("llama-server") is None or shutil.which("llama-embedding") is None:
        raise SystemExit("Missing llama.cpp tools. Install Termux package: pkg install llama-cpp")


def build_embeddings(
    conn: sqlite3.Connection,
    model_name: str,
    embed_texts: EmbedTexts,
    batch_size: int,
    rebuild: bool,
    limit: int | None,
) -> int:
    init_documents(conn)
    if rebuild:
        conn.execute("DELETE FROM card_embeddings WHERE model = ?", (model_name,))
        conn.commit()

    sql = """
        SELECT d.card_id, d.text, d.text_hash
        FROM card_semantic_documents d
        LEFT JOIN card_embeddings e
          ON e.card_id = d.card_id
         AND e.model = ?
         AND e.text_hash = d.text_hash
        WHERE e.card_id IS NULL
        ORDER BY d.card_id
    """
    params: list[object] = [model_name]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    docs = conn.execute(sql, params).fetchall()
    print(f"Missing embeddings for {model_name}: {len(docs)}", flush=True)
    if not docs:
        return 0
    print(f"Embedding up to {batch_size} card documents per request", flush=True)

    upsert = conn.execute
    written = 0

    for offset in range(0, len(docs), batch_size):
        batch = docs[offset : offset + batch_size]
        texts = [prefixed_document(row[1]) for row in batch]
        embeddings = run_llama_embeddings_with_retry(
            embed_texts,
            texts,
        )

        conn.execute("BEGIN")
        try:
            for row, embedding in zip(batch, embeddings):
                card_id, _text, text_hash = row
                upsert(
                    """
                    INSERT INTO card_embeddings(card_id, model, dimensions, embedding, text_hash, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(card_id, model) DO UPDATE SET
                      dimensions = excluded.dimensions,
                      embedding = excluded.embedding,
                      text_hash = excluded.text_hash,
                      updated_at = CURRENT_TIMESTAMP
                    """,
                    (card_id, model_name, len(embedding), vector_blob(embedding), text_hash),
                )
                written += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        if written % 256 == 0 or written == len(docs):
            print(f"Embedded {written}/{len(docs)}", flush=True)

    conn.execute("ANALYZE")
    conn.execute("PRAGMA optimize")
    return written


def ensure_model(conn: sqlite3.Connection, model_name: str) -> int:
    row = conn.execute(
        """
        SELECT dimensions
        FROM card_embeddings
        WHERE model = ?
        LIMIT 1
        """,
        (model_name,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"No embeddings found for model {model_name!r}. Run: python scripts/nomic_cards.py build")
    return int(row[0])


def search(
    conn: sqlite3.Connection,
    model_name: str,
    embed_texts: EmbedTexts,
    query: str,
    limit: int,
) -> list[sqlite3.Row]:
    dimensions = ensure_model(conn, model_name)
    query_embedding = embed_texts([prefixed_query(query)])[0]
    if len(query_embedding) != dimensions:
        raise ValueError(f"Query embedding has {len(query_embedding)} dimensions, expected {dimensions}")
    query_vector = array.array("f", query_embedding)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
          c.id,
          c.name,
          c.mana_cost,
          c.type,
          c.oracle_text,
          e.embedding
        FROM card_embeddings e
        JOIN cards c ON c.id = e.card_id
        WHERE e.model = ?
        """,
        (model_name,),
    ).fetchall()

    scored = []
    for row in rows:
        scored.append((dot_score(query_vector, row["embedding"]), row))
    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, row in scored[:limit]:
        result = dict(row)
        result["score"] = score
        results.append(result)
    return results


def print_results(rows: Iterable[dict]) -> None:
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
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help=f"GGUF model path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"SQLite model key. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--backend", choices=["server", "cli"], default="server")
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument(
        "--llama-batch-size",
        type=int,
        default=None,
        help="llama.cpp token batch size. Default: same as --ctx-size",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"llama-server host. Default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"llama-server port. Default: {DEFAULT_PORT}")
    parser.add_argument(
        "--server-url",
        default=None,
        help="Use an already-running llama-server instead of starting one.",
    )
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build Nomic embeddings into SQLite.")
    build.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Card documents per embedding request.",
    )
    build.add_argument("--limit", type=int, default=None, help="Limit rows for a test build.")
    build.add_argument("--rebuild", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search cards with Nomic embeddings.")
    search_parser.add_argument("query", nargs="+")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    database = args.database.resolve()
    llama_batch_size = args.llama_batch_size or args.ctx_size
    ensure_runtime(model_path)

    conn = connect(database)
    try:
        if args.backend == "cli":
            embed_texts = lambda texts: run_llama_embeddings_with_retry(
                lambda batch: run_llama_embeddings(
                    model_path,
                    batch,
                    ctx_size=args.ctx_size,
                    llama_batch_size=llama_batch_size,
                ),
                texts,
            )
            run_command(conn, args, embed_texts)
        elif args.server_url:
            server_url = args.server_url.rstrip("/")
            embed_texts = lambda texts: run_llama_embeddings_with_retry(
                lambda batch: request_server_embeddings(
                    server_url,
                    batch,
                    args.model_name,
                    timeout=args.request_timeout,
                ),
                texts,
            )
            run_command(conn, args, embed_texts)
        else:
            with managed_llama_server(
                model_path=model_path,
                host=args.host,
                port=args.port,
                ctx_size=args.ctx_size,
                llama_batch_size=llama_batch_size,
                model_name=args.model_name,
                request_timeout=args.request_timeout,
                startup_timeout=args.startup_timeout,
            ) as server_url:
                embed_texts = lambda texts: run_llama_embeddings_with_retry(
                    lambda batch: request_server_embeddings(
                        server_url,
                        batch,
                        args.model_name,
                        timeout=args.request_timeout,
                    ),
                    texts,
                )
                run_command(conn, args, embed_texts)
    finally:
        conn.close()


def run_command(conn: sqlite3.Connection, args: argparse.Namespace, embed_texts: EmbedTexts) -> None:
    if args.command == "build":
        count = build_embeddings(
            conn,
            model_name=args.model_name,
            embed_texts=embed_texts,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
            limit=args.limit,
        )
        print(f"Built {count} embeddings for {args.model_name}")
    elif args.command == "search":
        print_results(
            search(
                conn,
                model_name=args.model_name,
                embed_texts=embed_texts,
                query=" ".join(args.query),
                limit=args.limit,
            )
        )


if __name__ == "__main__":
    main()
