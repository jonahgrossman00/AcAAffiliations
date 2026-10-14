"""Streamlit front-end for author-affiliation extraction — paste a DOI or
article URL, get back every author's affiliation details below. Password-gated
so this can be shared as a link without letting anyone who gets it burn
OpenAI API credits.

Lets the user pick between the two retrieval engines in the project:
    - Full API Pipeline: new_retrieval.py's deterministic-APIs-first waterfall
      (Crossref/OpenAlex/PubMed, LLM only as a last resort to parse fetched text)
    - ChatGPT Web Search Only: extractor.py's pure web-search agent

No DB1/db1_doi_extractor.py connection is used here — this is standalone, for
sharing with people outside the network.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from new_retrieval import extract_affiliations as extract_full_pipeline
from extractor import extract_affiliations as extract_chatgpt_only

st.set_page_config(page_title="Affiliations AI", page_icon="🔎", layout="centered")

try:
    APP_PASSWORD = st.secrets.get("password", "acaaffiliations")
except Exception:  # noqa: BLE001 - no secrets.toml configured, fall back to the default
    APP_PASSWORD = "acaaffiliations"


def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True

    st.title("🔎 Affiliations AI")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

st.title("🔎 Affiliations AI")
st.caption("Paste a DOI or article URL below to extract every author's affiliation.")

method = st.radio(
    "Extraction method",
    ["Full API Pipeline", "ChatGPT Web Search Only"],
    horizontal=True,
    help=(
        "Full API Pipeline: tries Crossref/OpenAlex/PubMed first, only uses an LLM "
        "as a last resort (cheaper, faster, more reliable). "
        "ChatGPT Web Search Only: the model searches and browses the web itself for everything."
    ),
)

query = st.text_input(
    "DOI or article URL",
    placeholder="10.1063/5.0300419  or  https://doi.org/10.1063/5.0300419",
)
search = st.button("Search", type="primary")

if search:
    if not query.strip():
        st.warning("Enter a DOI or URL first.")
    else:
        with st.spinner("Extracting affiliations — this can take up to a couple of minutes..."):
            if method == "Full API Pipeline":
                data, sources, error = extract_full_pipeline(query.strip())
            else:
                target_url = query.strip() if query.strip().lower().startswith("http") else f"https://doi.org/{query.strip()}"
                data, sources, error = extract_chatgpt_only(target_url)

        if error:
            st.error(error)
        else:
            st.subheader(data.get("article_title") or "Untitled article")
            authors = data.get("authors", [])

            if not authors:
                st.info("No authors found.")

            for author in authors:
                with st.container(border=True):
                    st.markdown(f"**{author.get('name', 'Unknown')}**")
                    if author.get("email"):
                        st.caption(f"✉️ {author['email']}")

                    affiliations = author.get("affiliations") or []
                    if affiliations:
                        for i, aff in enumerate(affiliations, start=1):
                            label = f"Affiliation {i}" if len(affiliations) > 1 else "Affiliation"
                            parts = [p for p in (aff.get("institution"), aff.get("department"), aff.get("location")) if p]
                            st.write(f"**{label}:** " + (" — ".join(parts) if parts else "(none listed)"))
                    elif author.get("full_affiliation_text"):
                        st.write(author["full_affiliation_text"])
                    else:
                        st.write("_No affiliation data available._")

            if sources:
                with st.expander("Sources consulted this run"):
                    for s in sources:
                        st.markdown(f"- {s}")
