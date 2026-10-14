"""Run new_retrieval.py against the exact DOIs already stored in
ArticleWarehouse.dbo.AI_Article_Affiliations (extractor.py's results), match
authors by name, and write a simple side-by-side CSV of both pipelines'
affiliation text per author so it can be eyeballed.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

from rapidfuzz import fuzz

from db1_doi_extractor import TARGET_TABLE, WAREHOUSE_DB, get_connection
from new_retrieval import extract_affiliations

NAME_MATCH_THRESHOLD = 80

CSV_FIELDS = ["DOI", "Author Name", "New_Retrieval_Affiliation", "DB_Affiliation"]


def load_db_data():
    with get_connection(WAREHOUSE_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DOI, [Author Name], Affiliation FROM {TARGET_TABLE}")
        rows = cursor.fetchall()

    by_doi = {}
    for row in rows:
        name = (row[1] or "").strip()
        if not name:
            continue
        by_doi.setdefault(row.DOI, []).append({"Author Name": name, "Affiliation": row.Affiliation or ""})
    return by_doi


def process_doi(doi, index, total):
    print(f"[{index}/{total}] Extracting: {doi}", flush=True)
    try:
        data, _, error = extract_affiliations(doi)
    except Exception as exc:  # noqa: BLE001 - never let one DOI crash the whole batch
        error, data = f"Unexpected error: {exc}", None

    if error or not data:
        print(f"[{index}/{total}] FAILED: {doi} — {error}", flush=True)
        return doi, []

    authors = [
        {"Author Name": a.get("name", ""), "Affiliation": a.get("full_affiliation_text", "")}
        for a in data.get("authors", [])
        if a.get("name")
    ]
    print(f"[{index}/{total}] Done: {doi} — {len(authors)} author(s)", flush=True)
    return doi, authors


def match_name(name, candidates):
    best, best_score = None, 0
    for candidate in candidates:
        score = fuzz.token_sort_ratio(name.lower(), candidate.lower())
        if score > best_score:
            best, best_score = candidate, score
    return best if best_score >= NAME_MATCH_THRESHOLD else None


def build_rows(dois, nr_by_doi, db_by_doi):
    rows_out = []
    for doi in dois:
        nr_authors = nr_by_doi.get(doi, [])
        db_authors = db_by_doi.get(doi, [])
        nr_names = [a["Author Name"] for a in nr_authors]

        matched_db_names = set()
        for nr in nr_authors:
            match = match_name(nr["Author Name"], [d["Author Name"] for d in db_authors])
            db_affil = ""
            if match:
                matched_db_names.add(match)
                db_affil = next(d["Affiliation"] for d in db_authors if d["Author Name"] == match)
            rows_out.append(
                {
                    "DOI": doi,
                    "Author Name": nr["Author Name"],
                    "New_Retrieval_Affiliation": nr["Affiliation"],
                    "DB_Affiliation": db_affil,
                }
            )

        for db in db_authors:
            if db["Author Name"] in matched_db_names or match_name(db["Author Name"], nr_names):
                continue
            rows_out.append(
                {
                    "DOI": doi,
                    "Author Name": db["Author Name"],
                    "New_Retrieval_Affiliation": "",
                    "DB_Affiliation": db["Affiliation"],
                }
            )
    return rows_out


def main():
    parser = argparse.ArgumentParser(
        description="Compare new_retrieval.py against the DOIs already in ArticleWarehouse.dbo.AI_Article_Affiliations."
    )
    parser.add_argument("output", nargs="?", default="comparison_results.csv", help="Output comparison CSV")
    parser.add_argument("--workers", type=int, default=6, help="Number of DOIs to process concurrently (default: 6)")
    args = parser.parse_args()

    db_by_doi = load_db_data()
    dois = list(db_by_doi.keys())
    total = len(dois)
    print(f"Loaded {total} DOI(s) from {TARGET_TABLE}")

    nr_by_doi = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_doi, doi, i + 1, total): doi for i, doi in enumerate(dois)}
        for future in as_completed(futures):
            doi, authors = future.result()
            nr_by_doi[doi] = authors

    rows_out = build_rows(dois, nr_by_doi, db_by_doi)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
