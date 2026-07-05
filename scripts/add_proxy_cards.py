#!/usr/bin/env python3
"""Add card quantity lines to a proxy-order text file section."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST = ROOT / "proxy-orders" / "temmet-chat-buy-list-delta-2026-07-04.txt"
DEFAULT_SECTION = "Strong / must-buy additions discussed"

CARD_LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def normalize_card_name(name: str) -> str:
    return " ".join(name.strip().split()).casefold()


def section_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    return stripped.lstrip("#").strip()


def find_section(lines: list[str], section: str) -> tuple[int, int]:
    start = None
    for index, line in enumerate(lines):
        if section_heading(line) == section:
            start = index
            break

    if start is None:
        raise ValueError(f"Section not found: {section}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if section_heading(lines[index]) is not None:
            end = index
            break

    insert_at = end
    while insert_at > start + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1

    return insert_at, end


def existing_cards(lines: list[str]) -> set[str]:
    cards: set[str] = set()
    for line in lines:
        match = CARD_LINE_RE.match(line)
        if match:
            cards.add(normalize_card_name(match.group(2)))
    return cards


def add_cards(path: Path, section: str, quantity: int, cards: list[str]) -> tuple[list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    present = existing_cards(lines)
    insert_at, _ = find_section(lines, section)

    added: list[str] = []
    skipped: list[str] = []
    new_lines: list[str] = []

    for card in cards:
        clean = " ".join(card.strip().split())
        if not clean:
            continue
        normalized = normalize_card_name(clean)
        if normalized in present:
            skipped.append(clean)
            continue
        new_lines.append(f"{quantity} {clean}\n")
        present.add(normalized)
        added.append(clean)

    if new_lines:
        lines[insert_at:insert_at] = new_lines
        path.write_text("".join(lines), encoding="utf-8")

    return added, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cards", nargs="+", help="Card names to add")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_LIST,
        help=f"Proxy-order list to edit, default: {DEFAULT_LIST}",
    )
    parser.add_argument(
        "--section",
        default=DEFAULT_SECTION,
        help=f"Section heading to append to, default: {DEFAULT_SECTION!r}",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Quantity to add for each card, default: 1",
    )
    args = parser.parse_args()

    added, skipped = add_cards(args.file, args.section, args.quantity, args.cards)

    print(f"file: {args.file}")
    print(f"section: {args.section}")
    print(f"added: {len(added)}")
    for card in added:
        print(f"  + {card}")
    print(f"skipped_existing: {len(skipped)}")
    for card in skipped:
        print(f"  = {card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
