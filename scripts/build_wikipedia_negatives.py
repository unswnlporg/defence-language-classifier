#!/usr/bin/env python3
"""Build length-matched general and military Wikipedia negative passages."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defence_language_classifier.chunking import chunk_text, normalise_text, word_count  # noqa: E402


API_URL = "https://en.wikipedia.org/w/api.php"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.en"
HF_TOTAL_ROWS = 6_407_814
USER_AGENT = "defence-language-classifier-research/0.1 (academic dataset construction)"
MILITARY_QUERIES = (
    'incategory:"Military history"',
    'incategory:"Military operations"',
    'incategory:"Military strategy"',
    'incategory:"Armed forces"',
    'incategory:"Military terminology"',
)
MILITARY_TERMS = re.compile(
    r"\b(?:army|armed forces|battle|campaign|combat|defen[cs]e|force|military|navy|war|warfare|weapon)\b",
    re.IGNORECASE,
)
MILITARY_TITLE_TERMS = re.compile(
    r"(?:\bBattle of\b|\bWar\b|\bMilitary\b|\bArmy\b|\bNavy\b|\bAir Force\b|"
    r"\bArmed Forces\b|\bRegiment\b|\bBattalion\b|\bBrigade\b|\bArtillery\b|"
    r"\bInfantry\b|\bCavalry\b|\bDestroyer\b|\bFrigate\b|\bSubmarine\b|"
    r"\bFighter Squadron\b|\bBomber Squadron\b|\bmilitary officer\b|"
    r"\barmy officer\b|\bnaval officer\b|\bRoyal Navy officer\b)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("positive_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--general", type=int, default=446)
    parser.add_argument("--military", type=int, default=446)
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--request-delay", type=float, default=0.1)
    return parser.parse_args()


def api_get(params: Dict[str, object], delay: float) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            if delay:
                time.sleep(delay)
            return result
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 6:
                raise
            retry_after = error.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(2 ** (attempt + 1), 30)
            time.sleep(wait)
    raise RuntimeError("Wikipedia API retry loop exhausted")


def random_general_pages(batch_size: int, delay: float) -> List[dict]:
    result = api_get(
        {
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": min(batch_size, 20),
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
        },
        delay,
    )
    return result.get("query", {}).get("pages", [])


def military_titles(query: str, offset: int, delay: float) -> Tuple[List[str], Optional[int]]:
    result = api_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 0,
            "srlimit": 50,
            "sroffset": offset,
        },
        delay,
    )
    titles = [item["title"] for item in result.get("query", {}).get("search", [])]
    next_offset = result.get("continue", {}).get("sroffset")
    return titles, next_offset


def pages_for_titles(titles: List[str], delay: float) -> List[dict]:
    if not titles:
        return []
    result = api_get(
        {
            "action": "query",
            "titles": "|".join(titles[:20]),
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
        },
        delay,
    )
    return result.get("query", {}).get("pages", [])


def page_candidates(page: dict, negative_type: str, min_words: int, max_words: int) -> List[dict]:
    extract = normalise_text(page.get("extract", ""))
    if word_count(extract) < min_words:
        return []
    page_id = page.get("pageid")
    if not page_id:
        return []
    records = []
    for index, text in enumerate(chunk_text(extract, min_words, max_words)):
        count = word_count(text)
        if min_words <= count <= max_words:
            records.append(
                {
                    "document_id": f"wikipedia-{page_id}",
                    "chunk_index": index,
                    "text": text,
                    "word_count": count,
                    "title": page.get("title"),
                    "url": page.get("fullurl"),
                    "negative_type": negative_type,
                }
            )
    return records


def collect_general(required: int, min_words: int, max_words: int, delay: float) -> List[dict]:
    candidates: List[dict] = []
    seen_pages = set()
    attempts = 0
    while len(candidates) < required * 3 and attempts < 200:
        attempts += 1
        for page in random_general_pages(20, delay):
            page_id = page.get("pageid")
            if not page_id or page_id in seen_pages:
                continue
            seen_pages.add(page_id)
            extract = page.get("extract", "")
            # Reserve overtly military pages for the hard-negative group.
            if MILITARY_TERMS.search(f"{page.get('title', '')} {extract[:1000]}"):
                continue
            candidates.extend(page_candidates(page, "general_wikipedia", min_words, max_words))
    return candidates


def collect_military(required: int, min_words: int, max_words: int, delay: float) -> List[dict]:
    candidates: List[dict] = []
    seen_pages = set()
    for query in MILITARY_QUERIES:
        offset = 0
        while len(candidates) < required * 3:
            titles, next_offset = military_titles(query, offset, delay)
            for start in range(0, len(titles), 20):
                for page in pages_for_titles(titles[start : start + 20], delay):
                    page_id = page.get("pageid")
                    if not page_id or page_id in seen_pages:
                        continue
                    seen_pages.add(page_id)
                    candidates.extend(page_candidates(page, "military_wikipedia", min_words, max_words))
            if next_offset is None:
                break
            offset = next_offset
        if len(candidates) >= required * 3:
            break
    return candidates


def hf_rows(offset: int, length: int, delay: float) -> List[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": HF_DATASET,
            "config": HF_CONFIG,
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(f"{HF_ROWS_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
            if delay:
                time.sleep(delay)
            return [item["row"] for item in result.get("rows", [])]
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 6:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))
    raise RuntimeError("Hugging Face rows API retry loop exhausted")


def collect_hf_pools(
    general_required: int,
    military_required: int,
    min_words: int,
    max_words: int,
    delay: float,
    rng: random.Random,
) -> Tuple[List[dict], List[dict]]:
    general: List[dict] = []
    military: List[dict] = []
    seen_pages = set()
    seen_offsets = set()

    for _ in range(200):
        offset = rng.randrange(0, HF_TOTAL_ROWS - 100)
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        for row in hf_rows(offset, 100, delay):
            page_id = row.get("id")
            if not page_id or page_id in seen_pages:
                continue
            seen_pages.add(page_id)
            extract = row.get("text", "")
            # Use an explicit title-level signal for hard negatives. Broad body-text
            # keywords (for example, "force" or "campaign") create sports and biography
            # false positives and are deliberately not sufficient here.
            is_military = bool(MILITARY_TITLE_TERMS.search(row.get("title", "")))
            negative_type = "military_wikipedia" if is_military else "general_wikipedia"
            page = {
                "pageid": page_id,
                "title": row.get("title"),
                "fullurl": row.get("url"),
                "extract": extract,
            }
            candidates = page_candidates(page, negative_type, min_words, max_words)
            if is_military and len(military) < military_required * 3:
                military.extend(candidates)
            elif not is_military and len(general) < general_required * 3:
                general.extend(candidates)
        if len(general) >= general_required * 3 and len(military) >= military_required * 3:
            break

    if len(general) < general_required or len(military) < military_required:
        raise RuntimeError(
            f"Insufficient candidate pools: general={len(general)}, military={len(military)}"
        )
    return general, military


def load_target_lengths(path: Path, total: int, rng: random.Random) -> List[int]:
    lengths = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            lengths.append(json.loads(line)["word_count"])
    if total > len(lengths):
        raise ValueError(f"Requested {total} negatives but only {len(lengths)} positive lengths exist")
    rng.shuffle(lengths)
    return lengths[:total]


def length_match(candidates: Iterable[dict], targets: List[int], rng: random.Random) -> List[dict]:
    pool = list(candidates)
    rng.shuffle(pool)
    selected = []
    for target in targets:
        if not pool:
            raise RuntimeError("Candidate pool exhausted during length matching")
        best_index = min(range(len(pool)), key=lambda index: abs(pool[index]["word_count"] - target))
        row = pool.pop(best_index)
        row["matched_target_words"] = target
        selected.append(row)
    return selected


def stable_id(row: dict) -> str:
    payload = f"{row['document_id']}\0{row['chunk_index']}\0{row['text']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    total = args.general + args.military
    targets = load_target_lengths(args.positive_jsonl, total, rng)
    general_targets = targets[: args.general]
    military_targets = targets[args.general :]

    general_pool, military_pool = collect_hf_pools(
        args.general,
        args.military,
        args.min_words,
        args.max_words,
        args.request_delay,
        rng,
    )
    selected = length_match(general_pool, general_targets, rng) + length_match(
        military_pool, military_targets, rng
    )
    rng.shuffle(selected)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for row in selected:
            record = {
                "example_id": stable_id(row),
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "label": 0,
                "source_group": "wikipedia",
                "source_name": "enwiki",
                "negative_type": row["negative_type"],
                "word_count": row["word_count"],
                "matched_target_words": row["matched_target_words"],
                "title": row["title"],
                "url": row["url"],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "records_written": len(selected),
                "general": args.general,
                "military": args.military,
                "general_pool": len(general_pool),
                "military_pool": len(military_pool),
                "output": str(args.output_jsonl.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
