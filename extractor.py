"""Extracting author affiliations from an article URL using GPT-5.5's web browsing tool."""
import json
import re

import requests
from openai import OpenAI

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)

AFFILIATION_SCHEMA = {
    "name": "extracted_affiliations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "article_title": {"type": ["string", "null"]},
            "authors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": ["string", "null"]},
                        "affiliations": {
                            "type": "array",
                            "description": (
                                "One entry per distinct affiliation this author is listed "
                                "with (authors are often tagged with more than one)."
                            ),
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
                    "required": ["name", "email", "affiliations", "full_affiliation_text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["article_title", "authors"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You are an expert research assistant. Use the web search tool to open the exact "
    "article URL the user gives you and read its actual page content. Identify every "
    "listed author and, for each, extract their affiliation details exactly as stated "
    "on the page (institution, department, location/city/country, email). Do not "
    "invent, guess, or infer information that is not present on the page. If a field "
    "is not present for an author, use null for that field. List authors in the order "
    "they appear in the byline. Authors are frequently tagged with MORE THAN ONE "
    "affiliation (commonly shown as multiple superscript numbers or symbols next to "
    "their name) — include ALL of them as separate items in that author's "
    "'affiliations' array, in the order the markers appear. 'full_affiliation_text' "
    "should be the raw affiliation text for that author (covering every one of their "
    "affiliations) as close to verbatim as possible. If the publisher's page itself "
    "cannot be fetched (paywall, bot-blocking, timeout), do not give up — search the "
    "web for the same article by its title and/or DOI and look for the same "
    "author/affiliation metadata mirrored on an indexing or aggregator site (for "
    "example CrossRef, Semantic Scholar, PubMed, J-GLOBAL, or the authors' own "
    "institutional pages) and use that instead. If a verified title and/or author list "
    "is provided below (from a bibliographic registry such as CrossRef), treat it as "
    "ground truth for identifying the correct article: search using that exact title, "
    "confirm any page you land on is about that exact paper before extracting anything "
    "from it, and make sure your final author list matches the verified author names. "
    "Conference-proceedings metadata sometimes has an unrelated or mismatched "
    "conference/event name attached to the correct paper — do not let that field send "
    "you to a different, wrong paper from the same proceedings volume."
)


def _extract_doi(url: str):
    """Pull a bare DOI (e.g. '10.1063/5.0300419') out of a doi.org or publisher URL."""
    match = DOI_RE.search(url)
    if not match:
        return None
    return match.group(0).rstrip(".,;:)")


def _fetch_crossref_metadata(doi: str):
    """Look up the DOI's own registered title/authors from CrossRef, so the model has
    a verified anchor even when the publisher page is bot-blocked or its own metadata
    (e.g. conference/event name) is mismatched. Returns None on any failure."""
    try:
        resp = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "affiliations-ai/1.0 (mailto:noreply@example.com)"},
            timeout=10,
        )
        resp.raise_for_status()
        msg = resp.json().get("message", {})
    except Exception:  # noqa: BLE001 - CrossRef is a best-effort hint, never fatal
        return None

    title = (msg.get("title") or [None])[0]
    authors = []
    for a in msg.get("author", []):
        name = " ".join(part for part in (a.get("given"), a.get("family")) if part).strip()
        if name:
            authors.append(name)

    if not title and not authors:
        return None
    return {"title": title, "authors": authors}


def _build_user_message(url: str):
    message = f"Open this article URL and extract every author with their affiliations: {url}"

    doi = _extract_doi(url)
    metadata = _fetch_crossref_metadata(doi) if doi else None
    if not metadata:
        return message

    hints = []
    if metadata.get("title"):
        hints.append(f"Verified article title (from CrossRef): {metadata['title']}")
    if metadata.get("authors"):
        hints.append(
            "Verified author names (from CrossRef — affiliations were not available "
            "there, but use these names to confirm you have the complete, correct "
            "author list): " + "; ".join(metadata["authors"])
        )
    return message + "\n\n" + "\n".join(hints)


def _extract_sources(resp):
    """Pull every URL the model actually opened via the web_search tool during this
    run, in the order visited, deduplicated."""
    urls = []
    seen = set()
    for item in resp.output:
        if getattr(item, "type", None) != "web_search_call":
            continue
        action = getattr(item, "action", None)
        url = getattr(action, "url", None) if action else None
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def extract_affiliations(url: str, model: str = "gpt-5.5"):
    """Have the model open the URL itself (via web search/browsing) and extract
    structured author/affiliation data. Returns (data, sources, error_message)."""
    client = OpenAI()
    try:
        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            reasoning={"effort": "high"},
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(url)},
            ],
            text={"format": {"type": "json_schema", **AFFILIATION_SCHEMA}},
        )
    except Exception as exc:  # noqa: BLE001 - surface any OpenAI SDK error to the UI
        return None, [], f"OpenAI request failed: {exc}"

    try:
        data = json.loads(resp.output_text)
    except json.JSONDecodeError:
        return None, [], "Received a malformed response from the AI model."
    return data, _extract_sources(resp), None


def _normalize_tokens(name: str):
    return set(re.sub(r"[^a-z ]", "", name.lower()).split())


def match_author(target_name: str, authors: list):
    """Return authors whose name matches target_name, tolerant of order/initials/case."""
    target_tokens = _normalize_tokens(target_name)
    matches = []
    for author in authors:
        name_tokens = _normalize_tokens(author.get("name", ""))
        if target_tokens and target_tokens.issubset(name_tokens):
            matches.append(author)
        elif target_name.lower().strip() in author.get("name", "").lower():
            matches.append(author)
    return matches
