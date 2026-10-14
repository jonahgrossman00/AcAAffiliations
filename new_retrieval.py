"""Author-affiliation retrieval driven by deterministic bibliographic APIs first,
falling back to an LLM only to *parse* text this code has already fetched.

Retrieval order for a given DOI:
    1. DOI content negotiation (CSL-JSON via doi.org)  -> verified title + author names
    2. Crossref REST API                                -> backup title/authors, occasional affiliations
    3. OpenAlex                                          -> best source for structured affiliations (ROR-disambiguated)
    4. PubMed (if the DOI has a PMID)                    -> per-author affiliations from MEDLINE XML
    5. Publisher page fetched by this code, then handed
       to the LLM as raw text to parse (LLM never browses)
    6. Only for authors still unresolved: fall back to extractor.py's LLM web-search
       agent, which chooses what to search/open itself. Slower and less deterministic
       than steps 1-5, but can reach mirror/aggregator sites (Eureka, J-GLOBAL,
       ResearchGate, institutional pages) that neither the registries nor a single
       direct page fetch can.

For steps 1-5 the model is never given a search tool and never chooses a URL to
open — every page/record it sees was fetched by this code first. Every author
name any LLM stage returns (steps 5 and 6) is fuzzy-matched against the verified
registry author list (step 1) and dropped if it doesn't match anyone or carries
no actual affiliation content, so a mismatched/wrong-article page can't silently
inject the wrong people into the result.
"""
import json
import re
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup
from openai import OpenAI
from rapidfuzz import fuzz

from extractor import DOI_RE
from extractor import extract_affiliations as web_search_extract_affiliations

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
CROSSREF_HEADERS = {"User-Agent": "affiliations-ai/1.0 (mailto:noreply@example.com)"}

NAME_MATCH_THRESHOLD = 80  # rapidfuzz token_sort_ratio, 0-100
MAX_PAGE_TEXT_CHARS = 15000  # keep the LLM parse call cheap and focused
MIN_USABLE_PAGE_TEXT_CHARS = 200  # below this it's almost certainly an interstitial/redirect stub

_STOPWORDS = {"of", "and", "the", "a", "an", "in", "at", "for", "on", "de", "la", "et"}


def looks_like_real_text(text: str) -> bool:
    """Filters out garbled fragments like 'of , , , , ; , , , ; and ,' — OpenAlex
    (and occasionally Crossref) sometimes returns these when its own affiliation
    parser fails to extract real text: connector words and punctuation survive but
    the actual institution names don't. Requires at least one non-stopword word."""
    if not text:
        return False
    words = re.findall(r"[A-Za-z]{2,}", text)
    return any(w.lower() not in _STOPWORDS for w in words)

LLM_PARSE_SCHEMA = {
    "name": "parsed_affiliations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "authors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": ["string", "null"]},
                        "orcid": {"type": ["string", "null"]},
                        "affiliations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "institution": {"type": ["string", "null"]},
                                    "department": {"type": ["string", "null"]},
                                    "location": {"type": ["string", "null"]},
                                },
                                "required": ["institution", "department", "location"],
                                "additionalProperties": False,
                            },
                        },
                        "full_affiliation_text": {"type": "string"},
                    },
                    "required": ["name", "email", "orcid", "affiliations", "full_affiliation_text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["authors"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You are extracting author affiliation data from text that has already been "
    "fetched for you — you are NOT searching the web yourself. You will be given: "
    "(1) a verified article title and author list from a bibliographic registry "
    "(Crossref/OpenAlex/PubMed), and (2) raw page or metadata text pulled from one "
    "or more sources for that specific article. "
    "\n\n"
    "Your job is to extract, for each verified author, their affiliation details "
    "(institution, department, location, email) exactly as stated in the provided "
    "text. Do not invent, guess, or infer anything not present in the text. Use null "
    "for missing fields. "
    "\n\n"
    "Before extracting anything, confirm the provided text is actually describing "
    "the verified title/DOI given to you — if the text appears to describe a "
    "different paper (even one from the same issue, proceedings volume, or by an "
    "overlapping set of authors), return an empty author list rather than guessing. "
    "\n\n"
    "Your final author list must match the verified author names given to you — same "
    "people, same order. If the provided text is missing an author who appears in the "
    "verified list, include them with null affiliation fields rather than omitting "
    "them or substituting a different author. "
    "\n\n"
    "Authors are frequently tagged with more than one affiliation (multiple "
    "superscript markers) — include all of them, in marker order, in that author's "
    "affiliations array. full_affiliation_text should be the raw text for that "
    "author, as close to verbatim as possible."
)


def extract_doi(url_or_doi: str):
    match = DOI_RE.search(url_or_doi)
    if not match:
        return None
    return match.group(0).rstrip(".,;:)")


def _get_json(url, **kwargs):
    try:
        r = httpx.get(url, timeout=15, **kwargs)
        if r.status_code == 200:
            return r.json()
    except Exception:  # noqa: BLE001 - any registry is best-effort, never fatal
        pass
    return None


def get_csl_json(doi: str):
    return _get_json(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/vnd.citationstyles.csl+json"},
        follow_redirects=True,
    )


def get_crossref(doi: str):
    data = _get_json(f"https://api.crossref.org/works/{doi}", headers=CROSSREF_HEADERS)
    return data.get("message") if data else None


def get_openalex(doi: str):
    return _get_json(f"https://api.openalex.org/works/https://doi.org/{doi}")


def doi_to_pmid(doi: str):
    data = _get_json(
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
        params={"ids": doi, "format": "json"},
    )
    records = data.get("records", []) if data else []
    if records and records[0].get("pmid"):
        return records[0]["pmid"]
    return None


def get_pubmed_affiliations(pmid: str):
    """Returns {author_name: affiliation_text} parsed from MEDLINE XML, or {}."""
    try:
        r = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            timeout=15,
        )
        if r.status_code != 200:
            return {}
        root = ET.fromstring(r.text)
    except Exception:  # noqa: BLE001
        return {}

    results = {}
    for author in root.iter("Author"):
        last = author.findtext("LastName") or ""
        fore = author.findtext("ForeName") or author.findtext("Initials") or ""
        name = f"{fore} {last}".strip()
        if not name:
            continue
        affs = [a.text for a in author.findall("AffiliationInfo/Affiliation") if a.text and looks_like_real_text(a.text)]
        if affs:
            results[name] = "; ".join(affs)
    return results


def has_real_affiliation_content(author: dict) -> bool:
    """True if an LLM-returned author record actually contains affiliation text,
    not just a structurally non-empty 'affiliations' list whose entries are all
    null fields (the model sometimes emits one of these instead of []), or a
    garbled fragment with no real words in it."""
    if looks_like_real_text(author.get("full_affiliation_text") or ""):
        return True
    for aff in author.get("affiliations") or []:
        if any(looks_like_real_text((aff or {}).get(k) or "") for k in ("institution", "department", "location")):
            return True
    return False


def best_match(name: str, candidates: list, threshold: int = NAME_MATCH_THRESHOLD):
    """Fuzzy-match `name` against a list of verified names. Returns the matching
    verified name (so all sources converge on one canonical spelling) or None."""
    best_candidate, best_score = None, 0
    for candidate in candidates:
        score = fuzz.token_sort_ratio(name.lower(), candidate.lower())
        if score > best_score:
            best_candidate, best_score = candidate, score
    return best_candidate if best_score >= threshold else None


def fetch_publisher_page(doi: str):
    """Resolves the DOI redirect chain and fetches the final page ourselves.
    Returns (final_url, plain_text) or (final_url_or_None, None) on failure."""
    try:
        r = httpx.get(
            f"https://doi.org/{doi}", headers=BROWSER_HEADERS, follow_redirects=True, timeout=20
        )
    except Exception:  # noqa: BLE001
        return None, None

    final_url = str(r.url)
    if r.status_code != 200:
        return final_url, None

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Some publishers (e.g. Elsevier's linkinghub.elsevier.com) return a 200 with a
    # JS/meta-refresh interstitial ("Redirecting...") rather than real content, since
    # httpx doesn't execute JavaScript. There's nothing for the LLM to parse there.
    if len(text) < MIN_USABLE_PAGE_TEXT_CHARS:
        return final_url, None

    return final_url, text[:MAX_PAGE_TEXT_CHARS]


def llm_parse_from_text(verified_title, verified_authors, raw_text, model="gpt-5.5"):
    client = OpenAI()
    user_message = (
        f"Verified article title: {verified_title}\n"
        "Verified author names (extract affiliations only for these people): "
        + "; ".join(verified_authors)
        + "\n\n--- RAW FETCHED PAGE TEXT ---\n"
        + raw_text
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        text={"format": {"type": "json_schema", **LLM_PARSE_SCHEMA}},
    )
    data = json.loads(resp.output_text)
    return data.get("authors", [])


def extract_affiliations(doi_or_url: str, model: str = "gpt-5.5", use_web_search_fallback: bool = True):
    """Returns (data, sources, error) with the same shape as extractor.py's
    extract_affiliations: data = {"article_title", "authors": [...]}. Each author
    additionally carries "orcid" and "source" (which pipeline stage supplied it:
    openalex / crossref / pubmed / publisher_page_llm / web_search_llm / unavailable).

    use_web_search_fallback controls step 6 (extractor.py's web-search agent) for
    authors every deterministic source and the direct page fetch couldn't resolve.
    Disable it to keep runs fast/cheap/fully deterministic (e.g. large batch jobs)
    at the cost of leaving those authors as "unavailable"."""
    doi = extract_doi(doi_or_url)
    if not doi:
        return None, [], f"Could not find a DOI in '{doi_or_url}'"

    sources = [f"https://doi.org/{doi}"]

    # Step 1: verified ground truth (title + author names) — never skipped, never
    # produced by the LLM. Everything downstream is validated against this.
    csl = get_csl_json(doi)
    crossref = get_crossref(doi)

    verified_title = csl.get("title") if csl else None
    verified_authors = []
    if csl:
        for a in csl.get("author", []):
            name = " ".join(p for p in (a.get("given"), a.get("family")) if p).strip()
            if name:
                verified_authors.append(name)
    if not verified_authors and crossref:
        for a in crossref.get("author", []):
            name = " ".join(p for p in (a.get("given"), a.get("family")) if p).strip()
            if name:
                verified_authors.append(name)
    if not verified_title and crossref:
        verified_title = (crossref.get("title") or [None])[0]

    if not verified_authors:
        return None, sources, f"No registry metadata (CSL-JSON or Crossref) found for DOI {doi}"

    author_records = {}  # verified_name -> author dict

    # Step 2: OpenAlex — usually the best structured affiliation source.
    openalex = get_openalex(doi)
    if openalex:
        sources.append(f"https://api.openalex.org/works/https://doi.org/{doi}")
        for authorship in openalex.get("authorships", []):
            raw_name = (authorship.get("author") or {}).get("display_name", "")
            match = best_match(raw_name, verified_authors)
            if not match or match in author_records:
                continue
            institutions = authorship.get("institutions", []) or []
            raw_strings = [s for s in (authorship.get("raw_affiliation_strings") or []) if looks_like_real_text(s)]
            if not institutions and not raw_strings:
                continue
            orcid = ((authorship.get("author") or {}).get("orcid") or "").replace(
                "https://orcid.org/", ""
            ) or None
            author_records[match] = {
                "name": match,
                "email": None,
                "orcid": orcid,
                "affiliations": [
                    {"institution": inst.get("display_name"), "department": None, "location": inst.get("country_code")}
                    for inst in institutions
                ],
                "full_affiliation_text": "; ".join(raw_strings) or "; ".join(
                    inst.get("display_name", "") for inst in institutions
                ),
                "source": "openalex",
            }

    # Step 3: Crossref's own affiliation field, for whoever's still missing.
    if crossref:
        for a in crossref.get("author", []):
            raw_name = " ".join(p for p in (a.get("given"), a.get("family")) if p).strip()
            match = best_match(raw_name, verified_authors)
            if not match or match in author_records:
                continue
            affs = [x for x in (a.get("affiliation") or []) if looks_like_real_text(x.get("name", ""))]
            if not affs:
                continue
            author_records[match] = {
                "name": match,
                "email": None,
                "orcid": None,
                "affiliations": [{"institution": x.get("name"), "department": None, "location": None} for x in affs],
                "full_affiliation_text": "; ".join(x.get("name", "") for x in affs),
                "source": "crossref",
            }

    # Step 4: PubMed, only if biomedical (has a PMID) and gaps remain.
    missing = [n for n in verified_authors if n not in author_records]
    if missing:
        pmid = doi_to_pmid(doi)
        if pmid:
            pubmed_affs = get_pubmed_affiliations(pmid)
            if pubmed_affs:
                sources.append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                for raw_name, aff_text in pubmed_affs.items():
                    match = best_match(raw_name, verified_authors)
                    if not match or match in author_records:
                        continue
                    author_records[match] = {
                        "name": match,
                        "email": None,
                        "orcid": None,
                        "affiliations": [{"institution": aff_text, "department": None, "location": None}],
                        "full_affiliation_text": aff_text,
                        "source": "pubmed",
                    }

    # Step 5: last resort — fetch the publisher page ourselves and hand the LLM
    # the raw text to parse. The model never picks a URL; we already did.
    missing = [n for n in verified_authors if n not in author_records]
    if missing:
        page_url, page_text = fetch_publisher_page(doi)
        if page_url and page_url not in sources:
            sources.append(page_url)
        if page_text:
            try:
                llm_authors = llm_parse_from_text(verified_title, verified_authors, page_text, model=model)
            except Exception:  # noqa: BLE001 - LLM parse failure shouldn't kill the whole pipeline
                llm_authors = []
            for a in llm_authors:
                match = best_match(a.get("name", ""), verified_authors)
                if not match or match in author_records:
                    continue  # drop anything that doesn't map to a verified name
                if not has_real_affiliation_content(a):
                    continue  # LLM found nothing usable in the page text either
                a["name"] = match
                a["source"] = "publisher_page_llm"
                author_records[match] = a

    # Step 6: last resort — for authors every deterministic source and the direct
    # page fetch couldn't resolve, fall back to extractor.py's web-search agent
    # (the model chooses what to search/open, unlike every step above). Still
    # validated against the verified name list before anything is accepted.
    missing = [n for n in verified_authors if n not in author_records]
    if missing and use_web_search_fallback:
        try:
            ws_data, ws_sources, _ = web_search_extract_affiliations(
                f"https://doi.org/{doi}", model=model
            )
        except Exception:  # noqa: BLE001 - web-search fallback failure shouldn't kill the pipeline
            ws_data, ws_sources = None, []

        for s in ws_sources:
            if s not in sources:
                sources.append(s)

        if ws_data:
            for a in ws_data.get("authors", []):
                match = best_match(a.get("name", ""), verified_authors)
                if not match or match in author_records:
                    continue  # drop anything that doesn't map to a verified name
                if not has_real_affiliation_content(a):
                    continue  # web search found nothing usable either
                author_records[match] = {
                    "name": match,
                    "email": a.get("email"),
                    "orcid": None,
                    "affiliations": a.get("affiliations", []),
                    "full_affiliation_text": a.get("full_affiliation_text", ""),
                    "source": "web_search_llm",
                }

    # Step 7: assemble in verified order; anyone still missing gets a null record
    # rather than being silently dropped.
    final_authors = []
    for name in verified_authors:
        if name in author_records:
            final_authors.append(author_records[name])
        else:
            final_authors.append(
                {
                    "name": name,
                    "email": None,
                    "orcid": None,
                    "affiliations": [],
                    "full_affiliation_text": "",
                    "source": "unavailable",
                }
            )

    return {"article_title": verified_title, "authors": final_authors}, sources, None


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    target = sys.argv[1] if len(sys.argv) > 1 else input("DOI or URL: ").strip()
    result, used_sources, err = extract_affiliations(target)
    if err:
        print("ERROR:", err)
    else:
        print(json.dumps(result, indent=2))
    print("\nSOURCES:")
    for s in used_sources:
        print(" -", s)
