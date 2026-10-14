"""Pull pending DOIs from DB1's MatchingQueue, run them through the affiliation
extractor, and upsert the results into ArticleWarehouse.dbo.AI_Article_Affiliations.

The target table is created automatically on first run if it doesn't already
exist (see ensure_table_exists / CREATE_TABLE_SQL below).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyodbc
from dotenv import load_dotenv

load_dotenv()

from extractor import extract_affiliations

# Database that owns the unqualified tables in DOI_QUERY (MatchingQueue, ArticleScore).
QUERY_DB = "ArticleWarehouse"
WAREHOUSE_DB = "ArticleWarehouse"
TARGET_TABLE = "[ArticleWarehouse].[dbo].[AI_Article_Affiliations]"

# Exact query as used in SSMS against DB1 (the (SELECT ...) ... ORDER BY shape is
# valid T-SQL: a parenthesized query expression with a trailing ORDER BY on the
# outer expression), unchanged from what was verified there.
DOI_QUERY = """
DECLARE	@AuthorCount INT = 1
				 (SELECT TOP 50 mq.Name as DOI
				 FROM [MatchingQueue] mq
					INNER JOIN ArticleScore art  WITH (NOLOCK)
						ON mq.Name = art.DOI
                    INNER JOIN Matching..ArticleScoreDetail d   WITH (NOLOCK)
						ON art.PersonId = d.PersonId AND art.DOI = d.doi
					INNER JOIN AcademicAnalytics2026..vw_Core_InstitutionPerson cip
						ON art.personid = cip.PersonId
					LEFT JOIN AcademicAnalytics..Pub_ArticleFaculty paf
						ON art.personid = paf.personid AND art.DOI=paf.DOI AND MatchStatusId=1
					WHERE PAF.DOI IS NULL AND mq.UPDATEDBY IS NULL and COALESCE(AffiliationMeasurementScore,0)  = 0
						AND ((@AuthorCount IS NULL)
							OR (@AuthorCount = 1 AND AuthorCount <= 90)						)
					Group by mq.Name,PriorityFlag)
					ORDER BY PriorityFlag DESC,MAX(art.Totalscore) DESC;
"""

CREATE_TABLE_SQL = f"""
IF OBJECT_ID('dbo.AI_Article_Affiliations', 'U') IS NULL
BEGIN
    CREATE TABLE {TARGET_TABLE} (
        DOI           NVARCHAR(500)  NOT NULL,
        [Author Name] NVARCHAR(500)  NOT NULL,
        Affiliation   NVARCHAR(MAX)  NULL,
        Sources       NVARCHAR(MAX)  NULL,
        CONSTRAINT PK_AI_Article_Affiliations PRIMARY KEY (DOI, [Author Name])
    );
END
"""

UPSERT_SQL = f"""
MERGE {TARGET_TABLE} AS target
USING (SELECT ? AS DOI, ? AS AuthorName, ? AS Affiliation, ? AS Sources) AS source
    ON target.DOI = source.DOI AND target.[Author Name] = source.AuthorName
WHEN MATCHED THEN
    UPDATE SET Affiliation = source.Affiliation, Sources = source.Sources
WHEN NOT MATCHED THEN
    INSERT (DOI, [Author Name], Affiliation, Sources)
    VALUES (source.DOI, source.AuthorName, source.Affiliation, source.Sources);
"""


def get_connection(database: str):
    return pyodbc.connect(
        "Driver={SQL Server};"
        "Server=DB1;"
        f"Database={database};"
        "Trusted_Connection=yes;"
    )


def fetch_pending_dois():
    with get_connection(QUERY_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(DOI_QUERY)
        return [row.DOI for row in cursor.fetchall()]


def process_doi(doi, index, total):
    url = f"https://doi.org/{doi}"
    print(f"[{index}/{total}] Extracting: {doi}", flush=True)

    try:
        data, sources, error = extract_affiliations(url)
    except Exception as exc:  # noqa: BLE001 - never let one DOI crash the whole batch
        error = f"Unexpected error: {exc}"
        data, sources = None, []

    sources_str = "; ".join(sources)

    if error:
        print(f"[{index}/{total}] FAILED: {doi} — {error}", flush=True)
        return [{"DOI": doi, "Author Name": "", "Affiliation": f"ERROR: {error}", "Sources": sources_str}]

    authors = data.get("authors", []) if data else []
    if not authors:
        print(f"[{index}/{total}] Done: {doi} — no authors found", flush=True)
        return [{"DOI": doi, "Author Name": "", "Affiliation": "No authors found", "Sources": sources_str}]

    print(f"[{index}/{total}] Done: {doi} — {len(authors)} author(s) found", flush=True)
    return [
        {
            "DOI": doi,
            "Author Name": author.get("name", ""),
            "Affiliation": author.get("full_affiliation_text", ""),
            "Sources": sources_str,
        }
        for author in authors
    ]


def ensure_table_exists():
    with get_connection(WAREHOUSE_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        conn.commit()


def upsert_rows(rows):
    if not rows:
        return
    with get_connection(WAREHOUSE_DB) as conn:
        cursor = conn.cursor()
        for row in rows:
            cursor.execute(UPSERT_SQL, row["DOI"], row["Author Name"], row["Affiliation"], row["Sources"])
        conn.commit()


def main():
    parser = argparse.ArgumentParser(
        description="Extract affiliations for pending MatchingQueue DOIs and upsert into ArticleWarehouse."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of DOIs to process concurrently (default: 4)"
    )
    args = parser.parse_args()

    ensure_table_exists()

    dois = fetch_pending_dois()
    total = len(dois)
    print(f"Loaded {total} pending DOI(s) from MatchingQueue")

    all_rows = [None] * total
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(process_doi, doi, i + 1, total): i for i, doi in enumerate(dois)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            all_rows[idx] = future.result()

    flat_rows = [row for rows in all_rows for row in rows]
    upsert_rows(flat_rows)
    print(f"Upserted {len(flat_rows)} row(s) into {TARGET_TABLE}")


if __name__ == "__main__":
    main()
