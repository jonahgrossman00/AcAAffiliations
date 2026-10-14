"""Batch-run new_retrieval.py's deterministic-first pipeline against the pending
DOIs pulled from DB1's MatchingQueue (same query as db1_doi_extractor.py), and
save results to a CSV. This does NOT write to the database — it's for comparing
new_retrieval.py's output against extractor.py's / the existing warehouse table.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

from db1_doi_extractor import fetch_pending_dois
from new_retrieval import extract_affiliations

CSV_FIELDS = ["DOI", "Author Name", "Affiliation", "Source", "Sources"]


def process_doi(doi, index, total, use_web_search_fallback):
    print(f"[{index}/{total}] Extracting: {doi}", flush=True)

    try:
        data, sources, error = extract_affiliations(doi, use_web_search_fallback=use_web_search_fallback)
    except Exception as exc:  # noqa: BLE001 - never let one DOI crash the whole batch
        error = f"Unexpected error: {exc}"
        data, sources = None, []

    sources_str = "; ".join(sources)

    if error:
        print(f"[{index}/{total}] FAILED: {doi} — {error}", flush=True)
        return [{"DOI": doi, "Author Name": "", "Affiliation": f"ERROR: {error}", "Source": "", "Sources": sources_str}]

    authors = data.get("authors", []) if data else []
    if not authors:
        print(f"[{index}/{total}] Done: {doi} — no authors found", flush=True)
        return [{"DOI": doi, "Author Name": "", "Affiliation": "No authors found", "Source": "", "Sources": sources_str}]

    print(f"[{index}/{total}] Done: {doi} — {len(authors)} author(s) found", flush=True)
    return [
        {
            "DOI": doi,
            "Author Name": author.get("name", ""),
            "Affiliation": author.get("full_affiliation_text", ""),
            "Source": author.get("source", ""),
            "Sources": sources_str,
        }
        for author in authors
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run new_retrieval.py against pending MatchingQueue DOIs and save results to CSV."
    )
    parser.add_argument(
        "output", nargs="?", default="new_retrieval_results.csv", help="Output CSV file (default: new_retrieval_results.csv)"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of DOIs to process concurrently (default: 4)")
    parser.add_argument(
        "--no-web-search-fallback",
        action="store_true",
        help="Skip step 6 (extractor.py's web-search agent); unresolved authors stay 'unavailable' instead",
    )
    args = parser.parse_args()

    dois = fetch_pending_dois()
    total = len(dois)
    print(f"Loaded {total} pending DOI(s) from MatchingQueue")

    all_rows = [None] * total
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(process_doi, doi, i + 1, total, not args.no_web_search_fallback): i
            for i, doi in enumerate(dois)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            all_rows[idx] = future.result()

    flat_rows = [row for rows in all_rows for row in rows]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Wrote {len(flat_rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
