#!/usr/bin/env python3
"""Build topic-matched Wikipedia negatives for the Defence-language classifier.

For each frozen Defence passage, derive a short topic keyphrase locally (TF-IDF
over the Defence corpus only), send *only that keyphrase* to Wikipedia's public
search API (never the passage text), retrieve candidate articles, chunk them
into passages near the same word count, and rank candidates with:

    match_score = 0.7 * semantic_similarity + 0.2 * keyword_overlap + 0.1 * length_similarity

Each accepted pair inherits the split already frozen for its Defence document
in data/processed/dataset_splits.jsonl. Wikipedia articles are used at most
once across the whole dataset, so no split ever shares an article with
another split.

Runs in three phases so a walltime kill never loses finished work:
  1. Concurrent candidate-pool fetch (network-bound; cached per passage on disk).
  2. One batched embedding pass over every distinct candidate text.
  3. Serial score/select/commit, appending each decision to a progress log
     immediately. Re-running the script resumes from that log and from the
     per-passage pool cache instead of re-fetching or re-deciding anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defence_language_classifier.chunking import normalise_text, split_sentences, word_count  # noqa: E402
from defence_language_classifier.matching import (  # noqa: E402
    is_list_or_disambiguation,
    is_near_duplicate,
    keyword_overlap,
    length_similarity,
    match_score,
    sentence_windows,
    top_tfidf_terms,
    word_tolerance,
)

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "defence-language-classifier-research/0.1 (academic dataset construction; topic-matched negatives)"


# --- Wikipedia API (network-bound, safe to call from worker threads) -------


def api_get(params: dict, cache_dir: Path, delay: float) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    cache_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            cache_path.write_text(json.dumps(result), encoding="utf-8")
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


def search_titles(keyphrase: str, limit: int, cache_dir: Path, delay: float) -> list[dict]:
    result = api_get(
        {"action": "query", "list": "search", "srsearch": keyphrase, "srnamespace": 0, "srlimit": limit},
        cache_dir,
        delay,
    )
    return result.get("query", {}).get("search", [])


def fetch_extracts(page_ids: list[int], cache_dir: Path, delay: float) -> dict[int, dict]:
    pages: dict[int, dict] = {}
    for start in range(0, len(page_ids), 20):
        batch = page_ids[start : start + 20]
        result = api_get(
            {
                "action": "query",
                "pageids": "|".join(str(pid) for pid in batch),
                "prop": "extracts|info",
                "explaintext": 1,
                "exsectionformat": "plain",
                "inprop": "url",
            },
            cache_dir,
            delay,
        )
        for page in result.get("query", {}).get("pages", []):
            page_id = page.get("pageid")
            if page_id:
                pages[page_id] = page
    return pages


def candidate_windows_for_page(
    page: dict, defence_word_count: int, tolerance: int, min_words: int, max_extract_words: int
) -> list[tuple[str, int]]:
    extract = normalise_text(page.get("extract", ""))
    title = page.get("title", "")
    if word_count(extract) < min_words:
        return []
    if is_list_or_disambiguation(title, extract):
        return []
    # Bound window generation to the most topically representative part of the
    # article (its opening) instead of windowing the entire extract, which for
    # long articles produces a combinatorial number of near-duplicate windows.
    words = extract.split()
    if len(words) > max_extract_words:
        extract = " ".join(words[:max_extract_words])
    sentences = split_sentences(extract)
    if not sentences:
        return []
    counts = [word_count(s) for s in sentences]
    seen_text = set()
    windows = []
    for _start, _end, text, wc in sentence_windows(sentences, counts, defence_word_count, tolerance):
        if text in seen_text:
            continue
        seen_text.add(text)
        windows.append((text, wc))
    return windows


def build_query_variants(terms: list[str], title: str | None) -> list[str]:
    variants: list[str] = []
    if len(terms) >= 3:
        variants.append(" ".join(terms[:3]))
    if len(terms) >= 2:
        variants.append(" ".join(terms[:2]))
    for term in terms[:3]:
        variants.append(term)
    if title:
        variants.append(title)
    seen: list[str] = []
    for variant in variants:
        if variant and variant not in seen:
            seen.append(variant)
    return seen


def fetch_pool(
    defence_row: dict,
    terms: list[str],
    search_limit: int,
    min_words: int,
    max_extract_words: int,
    cache_dir: Path,
    delay: float,
    min_pool_size: int = 15,
    max_queries: int = 4,
) -> dict:
    """Network-bound: search + fetch extracts + chunk into candidate windows. No shared state.

    Tries multiple keyphrase variants (and the paper title) and unions their
    candidates rather than stopping at the first non-empty query, so the
    scorer sees a genuinely wide set of candidate articles per passage. Stops
    early once enough candidate pages have accumulated, to bound cost.
    """
    defence_wc = defence_row["word_count"]
    tolerance = word_tolerance(defence_wc)
    queries_tried = build_query_variants(terms, defence_row.get("title"))[:max_queries]

    pool: list[dict] = []
    seen_page_ids: set[int] = set()
    queries_used: list[str] = []
    error = None
    try:
        for query in queries_tried:
            hits = search_titles(query, search_limit, cache_dir, delay)
            page_ids = [hit["pageid"] for hit in hits if hit.get("pageid") and hit["pageid"] not in seen_page_ids]
            if not page_ids:
                continue
            pages = fetch_extracts(page_ids, cache_dir, delay)
            gained = False
            for page_id, page in pages.items():
                windows = candidate_windows_for_page(page, defence_wc, tolerance, min_words, max_extract_words)
                if windows:
                    gained = True
                seen_page_ids.add(page_id)
                for text, wc in windows:
                    pool.append(
                        {"page_id": page_id, "title": page.get("title"), "url": page.get("fullurl"), "text": text, "word_count": wc}
                    )
            if gained:
                queries_used.append(query)
            if len(seen_page_ids) >= min_pool_size:
                break
    except Exception as exc:  # noqa: BLE001 - one bad passage must not kill the whole run
        error = f"{type(exc).__name__}: {exc}"
    query_used = "; ".join(queries_used) if queries_used else None

    return {"example_id": defence_row["example_id"], "queries_tried": queries_tried, "query_used": query_used, "pool": pool, "error": error}


# --- IO helpers --------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def stable_id(*parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values)
    return {
        "min": float(arr.min()),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defence", type=Path, default=Path("data/processed/defence_passages.jsonl"))
    parser.add_argument("--dataset-splits", type=Path, default=Path("data/processed/dataset_splits.jsonl"))
    parser.add_argument("--negatives-output", type=Path, default=Path("data/processed/topic_matched_wikipedia_negatives.jsonl"))
    parser.add_argument("--pairs-output", type=Path, default=Path("data/processed/topic_matched_pairs.jsonl"))
    parser.add_argument("--combined-output", type=Path, default=Path("data/processed/topic_matched_dataset_splits.jsonl"))
    parser.add_argument("--manifest-output", type=Path, default=Path("data/processed/topic_matched_split_manifest.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/topic_matched"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/topic_matching"))
    parser.add_argument("--progress", type=Path, default=None, help="Defaults to <cache-dir>/decisions.jsonl")
    parser.add_argument("--retrieval-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=200)
    parser.add_argument("--max-extract-words", type=int, default=400, help="Cap on how much of each article's opening text is windowed into candidates")
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument("--semantic-floor", type=float, default=0.40)
    parser.add_argument("--near-dup-threshold", type=float, default=0.5)
    parser.add_argument("--keyphrase-terms", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=10, help="Concurrent Wikipedia API fetch threads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weak-match-quantile", type=float, default=0.10)
    parser.add_argument("--finalize-only", action="store_true", help="Skip fetch/score, just rebuild outputs from the progress log")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    pool_cache_dir = args.cache_dir / "pools"
    pool_cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress or (args.cache_dir / "decisions.jsonl")

    defence_rows = read_jsonl(args.defence)
    split_rows = read_jsonl(args.dataset_splits)
    doc_split = {row["document_id"]: row["split"] for row in split_rows if row["source_group"] == "defence"}
    missing_split = [row["document_id"] for row in defence_rows if row["document_id"] not in doc_split]
    if missing_split:
        raise ValueError(
            f"{len(set(missing_split))} Defence documents have no frozen split in "
            f"{args.dataset_splits}; the split assignment must stay immutable."
        )

    # --- Resume state from the progress log --------------------------------
    decisions = read_jsonl(progress_path)
    resolved_ids = {d["example_id"] for d in decisions}
    pairs = [d for d in decisions if d["type"] == "pair"]
    unmatched = [d for d in decisions if d["type"] == "unmatched"]
    used_page_ids: set[int] = {p["wikipedia_page_id"] for p in pairs}
    accepted_wiki_texts: list[str] = [p["wikipedia_text"] for p in pairs]
    print(f"Resumed: {len(pairs)} pairs, {len(unmatched)} unmatched already decided ({len(resolved_ids)} passages resolved).")

    rng = random.Random(args.seed)
    order = list(range(len(defence_rows)))
    rng.shuffle(order)
    remaining = [i for i in order if defence_rows[i]["example_id"] not in resolved_ids]

    if not args.finalize_only and remaining:
        tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.9, stop_words="english")
        tfidf_matrix = tfidf.fit_transform([row["text"] for row in defence_rows])
        feature_names = tfidf.get_feature_names_out()

        terms_by_index = {
            i: top_tfidf_terms(feature_names, tfidf_matrix.getrow(i), top_k=args.keyphrase_terms) for i in remaining
        }

        # --- Phase 1: concurrent candidate-pool fetch, cached per passage --
        to_fetch = [i for i in remaining if not (pool_cache_dir / f"{defence_rows[i]['example_id']}.json").exists()]
        print(f"Phase 1: fetching candidate pools for {len(to_fetch)}/{len(remaining)} passages ({args.workers} workers)...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    fetch_pool,
                    defence_rows[i],
                    terms_by_index[i],
                    args.search_limit,
                    args.min_words,
                    args.max_extract_words,
                    args.cache_dir,
                    args.request_delay,
                ): i
                for i in to_fetch
                if terms_by_index[i]
            }
            done_count = 0
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                example_id = defence_rows[index]["example_id"]
                (pool_cache_dir / f"{example_id}.json").write_text(json.dumps(result), encoding="utf-8")
                done_count += 1
                if done_count % 50 == 0:
                    print(f"  fetched {done_count}/{len(to_fetch)}")
        print("Phase 1 complete.")

        # --- Phase 2: one batched embedding pass over all distinct texts ---
        print("Phase 2: batch-encoding candidate texts...")
        text_set: set[str] = {defence_rows[i]["text"] for i in remaining}
        pool_by_index: dict[int, dict] = {}
        for i in remaining:
            example_id = defence_rows[i]["example_id"]
            terms = terms_by_index[i]
            pool_path = pool_cache_dir / f"{example_id}.json"
            if not terms:
                pool_by_index[i] = {"example_id": example_id, "queries_tried": [], "query_used": None, "pool": [], "error": "no_keyphrase_terms"}
                continue
            result = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else fetch_pool(
                defence_rows[i], terms, args.search_limit, args.min_words, args.max_extract_words, args.cache_dir, args.request_delay
            )
            pool_by_index[i] = result
            text_set.update(c["text"] for c in result["pool"])

        all_texts = sorted(text_set)
        embedding_cache_path = args.cache_dir / f"embeddings_{args.retrieval_encoder.replace('/', '_')}.npz"
        embedding_by_text: dict[str, np.ndarray] = {}
        if embedding_cache_path.exists():
            cached = np.load(embedding_cache_path, allow_pickle=True)
            embedding_by_text = dict(zip(cached["texts"].tolist(), cached["vectors"]))
        to_encode = [t for t in all_texts if t not in embedding_by_text]
        print(f"  {len(all_texts) - len(to_encode)}/{len(all_texts)} texts already embedded and cached; encoding {len(to_encode)} new.")
        if to_encode:
            encoder = SentenceTransformer(args.retrieval_encoder)
            new_embeddings = encoder.encode(
                to_encode, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
            for text, vec in zip(to_encode, new_embeddings):
                embedding_by_text[text] = vec
            np.savez_compressed(
                embedding_cache_path,
                texts=np.array(list(embedding_by_text.keys()), dtype=object),
                vectors=np.array(list(embedding_by_text.values())),
            )
        print(f"Phase 2 complete ({len(all_texts)} distinct texts available).")

        # --- Phase 3: serial score/select/commit, checkpointed -------------
        print("Phase 3: scoring and selecting matches...")
        for rank, index in enumerate(remaining):
            defence_row = defence_rows[index]
            example_id = defence_row["example_id"]
            defence_text = defence_row["text"]
            defence_wc = defence_row["word_count"]
            result = pool_by_index[index]

            if result.get("error") and not result["pool"]:
                decision = {"type": "unmatched", "example_id": example_id, "reason": "fetch_error", "detail": result["error"]}
                unmatched.append(decision)
                append_jsonl(progress_path, decision)
                continue
            if not result["pool"]:
                decision = {"type": "unmatched", "example_id": example_id, "reason": "no_candidates", "queries": result["queries_tried"]}
                unmatched.append(decision)
                append_jsonl(progress_path, decision)
                continue

            defence_embedding = embedding_by_text[defence_text]
            scored = []
            for candidate in result["pool"]:
                if candidate["page_id"] in used_page_ids:
                    continue
                if is_near_duplicate(defence_text, candidate["text"], args.near_dup_threshold):
                    continue
                if any(is_near_duplicate(candidate["text"], existing, args.near_dup_threshold) for existing in accepted_wiki_texts):
                    continue
                semantic = float(embedding_by_text[candidate["text"]] @ defence_embedding)
                keyword = keyword_overlap(defence_text, candidate["text"])
                length = length_similarity(defence_wc, candidate["word_count"])
                score = match_score(semantic, keyword, length)
                scored.append({**candidate, "semantic_similarity": semantic, "keyword_overlap": keyword, "length_similarity": length, "match_score": score})

            scored.sort(key=lambda item: item["match_score"], reverse=True)
            best = next((item for item in scored if item["semantic_similarity"] >= args.semantic_floor), None)

            if best is None:
                decision = {
                    "type": "unmatched",
                    "example_id": example_id,
                    "reason": "below_semantic_floor",
                    "best_semantic_similarity": scored[0]["semantic_similarity"] if scored else None,
                    "candidates_considered": len(scored),
                }
                unmatched.append(decision)
                append_jsonl(progress_path, decision)
                continue

            used_page_ids.add(best["page_id"])
            accepted_wiki_texts.append(best["text"])
            split = doc_split[defence_row["document_id"]]
            wiki_example_id = stable_id(f"wikipedia-{best['page_id']}", best["text"])
            decision = {
                "type": "pair",
                "pair_id": f"pair-{len(pairs):04d}",
                "example_id": example_id,
                "defence_example_id": example_id,
                "defence_document_id": defence_row["document_id"],
                "wikipedia_example_id": wiki_example_id,
                "defence_text": defence_text,
                "wikipedia_text": best["text"],
                "topic_query": result["query_used"],
                "wikipedia_title": best["title"],
                "wikipedia_url": best["url"],
                "wikipedia_page_id": best["page_id"],
                "semantic_similarity": best["semantic_similarity"],
                "keyword_overlap": best["keyword_overlap"],
                "length_similarity": best["length_similarity"],
                "match_score": best["match_score"],
                "defence_word_count": defence_wc,
                "wikipedia_word_count": best["word_count"],
                "candidates_considered": len(scored),
                "split": split,
            }
            pairs.append(decision)
            append_jsonl(progress_path, decision)
            if len(pairs) % 50 == 0:
                print(f"  matched {len(pairs)} pairs so far ({len(unmatched)} unmatched)")
        print("Phase 3 complete.")

    # --- Finalize: rebuild all outputs from the full decision set ----------
    negatives = []
    for pair in pairs:
        negatives.append(
            {
                "example_id": pair["wikipedia_example_id"],
                "document_id": f"wikipedia-{pair['wikipedia_page_id']}",
                "chunk_index": 0,
                "text": pair["wikipedia_text"],
                "label": 0,
                "source_group": "wikipedia",
                "source_name": "enwiki-topic-matched",
                "negative_type": "topic_matched",
                "word_count": pair["wikipedia_word_count"],
                "title": pair["wikipedia_title"],
                "url": pair["wikipedia_url"],
                "matched_defence_example_id": pair["defence_example_id"],
                "topic_query": pair["topic_query"],
                "semantic_similarity": pair["semantic_similarity"],
                "keyword_overlap": pair["keyword_overlap"],
                "length_similarity": pair["length_similarity"],
                "match_score": pair["match_score"],
                "split": pair["split"],
            }
        )
    write_jsonl(args.negatives_output, negatives)
    write_jsonl(args.pairs_output, pairs)

    combined = [{**row, "split": doc_split[row["document_id"]]} for row in defence_rows]
    combined.extend(negatives)
    combined.sort(key=lambda row: (row["split"], row["document_id"], row.get("chunk_index", 0)))
    write_jsonl(args.combined_output, combined)

    manifest = {
        "seed": args.seed,
        "retrieval_encoder": args.retrieval_encoder,
        "semantic_floor": args.semantic_floor,
        "near_dup_threshold": args.near_dup_threshold,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "requested_pairs": len(defence_rows),
        "accepted_pairs": len(pairs),
        "unmatched": len(unmatched),
        "splits_inherited_from": str(args.dataset_splits),
        "summary": {split: {"pairs": sum(1 for p in pairs if p["split"] == split)} for split in ("train", "validation", "test")},
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    match_scores = [p["match_score"] for p in pairs]
    weak_cutoff = float(np.quantile(match_scores, args.weak_match_quantile)) if match_scores else None
    weak_matches = [p for p in pairs if weak_cutoff is not None and p["match_score"] <= weak_cutoff]
    write_jsonl(args.report_dir / "weak_matches.jsonl", weak_matches)

    audit = {
        "requested_pairs": len(defence_rows),
        "accepted_pairs": len(pairs),
        "shortfall": len(defence_rows) - len(pairs),
        "resolved": len(pairs) + len(unmatched),
        "unmatched_reasons": {reason: sum(1 for u in unmatched if u["reason"] == reason) for reason in {u["reason"] for u in unmatched}},
        "distributions": {
            "semantic_similarity": quantiles([p["semantic_similarity"] for p in pairs]),
            "keyword_overlap": quantiles([p["keyword_overlap"] for p in pairs]),
            "length_similarity": quantiles([p["length_similarity"] for p in pairs]),
            "match_score": quantiles(match_scores),
            "word_count_diff": quantiles([abs(p["defence_word_count"] - p["wikipedia_word_count"]) for p in pairs]),
        },
        "weak_match_quantile": args.weak_match_quantile,
        "weak_match_score_cutoff": weak_cutoff,
        "weak_match_count": len(weak_matches),
        "split_counts": manifest["summary"],
    }
    (args.report_dir / "matching_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.report_dir / "unmatched.jsonl", unmatched)

    print(json.dumps(audit, indent=2))
    if len(pairs) + len(unmatched) < len(defence_rows):
        print(
            f"\nNOT DONE: {len(pairs) + len(unmatched)}/{len(defence_rows)} passages resolved. "
            "Re-run the same command to resume from the progress log.",
            file=sys.stderr,
        )
    elif len(pairs) < len(defence_rows):
        print(
            f"\nWARNING: only matched {len(pairs)}/{len(defence_rows)} Defence passages. "
            "See reports/topic_matched/unmatched.jsonl for reasons. Not padding the shortfall.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
