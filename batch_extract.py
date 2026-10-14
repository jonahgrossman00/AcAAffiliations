"""Batch-extract author affiliations for a list of articles defined in a JSON file.

Input JSON is a list of objects like:
    [{"Article Publisher": "Institute of Physics", "DOI": "https://doi.org/10.1088/..."}]

Output CSV has one row per author: Article Publisher, DOI, Author Name, Affiliation, Sources.
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

from extractor import extract_affiliations

CSV_FIELDS = ["Article Publisher", "DOI", "Author Name", "Affiliation", "Sources"]


def load_articles(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_article(article, index, total):
    publisher = article.get("Article Publisher", "")
    doi_url = article.get("DOI", "")
    print(f"[{index}/{total}] Extracting: {publisher} — {doi_url}", flush=True)

    if not doi_url:
        print(f"[{index}/{total}] SKIPPED (no DOI/URL): {publisher}", flush=True)
        return [
            {
                "Article Publisher": publisher,
                "DOI": doi_url,
                "Author Name": "",
                "Affiliation": "",
                "Sources": "ERROR: no DOI/URL provided",
            }
        ]

    try:
        data, sources, error = extract_affiliations(doi_url)
    except Exception as exc:  # noqa: BLE001 - never let one article crash the whole batch
        error = f"Unexpected error: {exc}"
        data, sources = None, []

    sources_str = "; ".join(sources)

    if error:
        print(f"[{index}/{total}] FAILED: {publisher} — {error}", flush=True)
        return [
            {
                "Article Publisher": publisher,
                "DOI": doi_url,
                "Author Name": "",
                "Affiliation": "",
                "Sources": f"ERROR: {error}",
            }
        ]

    authors = data.get("authors", [])
    if not authors:
        print(f"[{index}/{total}] Done: {publisher} — no authors found", flush=True)
        return [
            {
                "Article Publisher": publisher,
                "DOI": doi_url,
                "Author Name": "",
                "Affiliation": "No authors found",
                "Sources": sources_str,
            }
        ]

    print(f"[{index}/{total}] Done: {publisher} — {len(authors)} author(s) found", flush=True)
    return [
        {
            "Article Publisher": publisher,
            "DOI": doi_url,
            "Author Name": author.get("name", ""),
            "Affiliation": author.get("full_affiliation_text", ""),
            "Sources": sources_str,
        }
        for author in authors
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Batch-extract author affiliations from a JSON list of articles."
    )
    parser.add_argument(
        "input", nargs="?", default="articles.json", help="Input JSON file (default: articles.json)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="affiliations_results.csv",
        help="Output CSV file (default: affiliations_results.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of articles to process concurrently (default: 4)",
    )
    args = parser.parse_args()

    articles = load_articles(args.input)
    total = len(articles)
    print(f"Loaded {total} article(s) from {args.input}")

    all_rows = [None] * total
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(process_article, article, i + 1, total): i
            for i, article in enumerate(articles)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            all_rows[idx] = future.result()

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rows in all_rows:
            writer.writerows(rows)

    total_rows = sum(len(rows) for rows in all_rows)
    print(f"Wrote {total_rows} row(s) to {args.output}")


if __name__ == "__main__":
    main()
