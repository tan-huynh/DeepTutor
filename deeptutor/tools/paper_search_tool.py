"""
Multi-source academic paper search.

Sources (tried in priority order when multi-source is enabled):
  1. arXiv    – preprints, free, no auth
  2. Semantic Scholar (S2) – citation graph, open-access PDFs, no auth required
  3. PubMed   – biomedical literature, NCBI E-utilities, no auth for basic use

Each client returns a normalised ``PaperRecord`` dict:
    {
        "title":    str,
        "authors":  list[str],
        "year":     int | None,
        "abstract": str,
        "url":      str,
        "doi":      str,          # empty if unknown
        "source":   str,          # "arxiv" | "semantic_scholar" | "pubmed"
        "arxiv_id": str,          # arXiv ID or empty
        "pmid":     str,          # PubMed ID or empty
        "s2_id":    str,          # Semantic Scholar corpusId or empty
        "published": str,         # ISO-8601 date or empty
        "citations": int,         # citation count or 0
    }
"""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
import urllib.request
import urllib.error
import json
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REQUEST_TIMEOUT_S = 20
_MAX_RETRIES = 2
_RETRY_DELAY_S = 2.0
_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# ---------------------------------------------------------------------------
# Shared HTTP helper (stdlib only — no extra deps)
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = _REQUEST_TIMEOUT_S) -> dict[str, Any] | list | None:
    """Synchronous JSON GET. Returns parsed JSON or None on error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DeepTutor/1.0 (research tool)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("HTTP %s for %s", exc.code, url)
        return None
    except Exception as exc:
        logger.warning("GET failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# arXiv client
# ---------------------------------------------------------------------------

class ArxivClient:
    """Search arXiv using the arxiv library (if installed) or Atom feed API."""

    _USE_LIB: bool | None = None  # lazy check

    @classmethod
    def _has_lib(cls) -> bool:
        if cls._USE_LIB is None:
            try:
                import arxiv  # noqa: F401
                cls._USE_LIB = True
            except ImportError:
                cls._USE_LIB = False
        return bool(cls._USE_LIB)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        years_limit: int | None = 3,
        sort_by: str = "relevance",
    ) -> list[dict[str, Any]]:
        if self._has_lib():
            return await self._search_via_lib(query, max_results, years_limit, sort_by)
        return await asyncio.to_thread(self._search_via_atom, query, max_results, years_limit)

    async def _search_via_lib(
        self,
        query: str,
        max_results: int,
        years_limit: int | None,
        sort_by: str,
    ) -> list[dict[str, Any]]:
        import arxiv  # type: ignore[import]

        sort_map = {
            "date": arxiv.SortCriterion.SubmittedDate,
            "relevance": arxiv.SortCriterion.Relevance,
        }
        client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=_MAX_RETRIES)
        search = arxiv.Search(
            query=query,
            max_results=min(max_results * 2, 30),
            sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(lambda: list(client.results(search))),
                timeout=_REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("arXiv lib search timed out for: %s", query)
            return []
        except Exception as exc:
            logger.warning("arXiv lib error for '%s': %s", query, exc)
            return []

        current_year = datetime.now().year
        papers: list[dict[str, Any]] = []
        for r in raw:
            if years_limit and (current_year - r.published.year) > years_limit:
                continue
            arxiv_id = r.entry_id.split("/")[-1]
            if "v" in arxiv_id:
                arxiv_id = arxiv_id.split("v")[0]
            papers.append(self._normalise(r, arxiv_id))
            if len(papers) >= max_results:
                break
        return papers

    def _normalise(self, r: Any, arxiv_id: str) -> dict[str, Any]:
        return {
            "title": r.title or "",
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "abstract": " ".join((r.summary or "").split()),
            "url": r.entry_id,
            "doi": r.doi or "",
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "pmid": "",
            "s2_id": "",
            "published": r.published.isoformat(),
            "citations": 0,
        }

    def _search_via_atom(
        self,
        query: str,
        max_results: int,
        years_limit: int | None,
    ) -> list[dict[str, Any]]:
        """Fallback: arXiv Atom feed (no library needed)."""
        import xml.etree.ElementTree as ET

        encoded = urllib.parse.quote(query)
        url = (
            f"https://export.arxiv.org/api/query?search_query=all:{encoded}"
            f"&start=0&max_results={min(max_results * 2, 30)}&sortBy=relevance&sortOrder=descending"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DeepTutor/1.0"})
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                xml_bytes = resp.read()
        except Exception as exc:
            logger.warning("arXiv Atom fallback failed: %s", exc)
            return []

        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return []

        current_year = datetime.now().year
        papers: list[dict[str, Any]] = []
        for entry in root.findall("a:entry", ns):
            published_str = (entry.findtext("a:published", "", ns) or "").strip()
            year: int | None = None
            if published_str:
                try:
                    year = int(published_str[:4])
                except ValueError:
                    pass
            if years_limit and year and (current_year - year) > years_limit:
                continue

            entry_id = (entry.findtext("a:id", "", ns) or "").strip()
            arxiv_id = entry_id.split("/")[-1]
            if "v" in arxiv_id:
                arxiv_id = arxiv_id.split("v")[0]

            authors = [
                (a.findtext("a:name", "", ns) or "").strip()
                for a in entry.findall("a:author", ns)
            ]
            abstract = " ".join((entry.findtext("a:summary", "", ns) or "").split())

            papers.append({
                "title": (entry.findtext("a:title", "", ns) or "").strip(),
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "url": entry_id,
                "doi": "",
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "pmid": "",
                "s2_id": "",
                "published": published_str,
                "citations": 0,
            })
            if len(papers) >= max_results:
                break
        return papers


# ---------------------------------------------------------------------------
# Semantic Scholar client
# ---------------------------------------------------------------------------

class SemanticScholarClient:
    """
    Search Semantic Scholar via the public Graph API.
    No API key required for ≤100 req/5min.
    Docs: https://api.semanticscholar.org/api-docs/graph
    """

    _FIELDS = "title,authors,year,abstract,externalIds,url,openAccessPdf,citationCount,publicationDate"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        years_limit: int | None = 3,
    ) -> list[dict[str, Any]]:
        encoded = urllib.parse.quote(query)
        url = (
            f"{_S2_BASE}/paper/search"
            f"?query={encoded}&limit={min(max_results * 2, 20)}&fields={self._FIELDS}"
        )
        data = await asyncio.to_thread(_http_get, url)
        if not isinstance(data, dict):
            return []

        current_year = datetime.now().year
        papers: list[dict[str, Any]] = []
        for item in data.get("data", []):
            year: int | None = item.get("year")
            if years_limit and year and (current_year - year) > years_limit:
                continue
            papers.append(self._normalise(item))
            if len(papers) >= max_results:
                break
        return papers

    def _normalise(self, item: dict[str, Any]) -> dict[str, Any]:
        ext = item.get("externalIds") or {}
        authors = [
            a.get("name", "") for a in (item.get("authors") or [])
        ]
        pub_date = item.get("publicationDate") or ""
        doi = ext.get("DOI") or ext.get("doi") or ""
        arxiv_id = ext.get("ArXiv") or ext.get("arxiv") or ""
        pmid = str(ext.get("PubMed") or ext.get("PMID") or "")

        pdf_info = item.get("openAccessPdf") or {}
        url = pdf_info.get("url") or item.get("url") or ""
        if not url and arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"

        return {
            "title": item.get("title") or "",
            "authors": authors,
            "year": item.get("year"),
            "abstract": item.get("abstract") or "",
            "url": url,
            "doi": doi,
            "source": "semantic_scholar",
            "arxiv_id": arxiv_id,
            "pmid": pmid,
            "s2_id": item.get("paperId") or item.get("corpusId") or "",
            "published": pub_date,
            "citations": int(item.get("citationCount") or 0),
        }


# ---------------------------------------------------------------------------
# PubMed client (NCBI E-utilities, no auth required)
# ---------------------------------------------------------------------------

class PubMedClient:
    """
    Search PubMed via NCBI E-utilities.
    Biomedical literature only. No API key for ≤3 req/sec.
    Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    """

    _TOOL = "DeepTutor"
    _EMAIL = "research@deeptutor.ai"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        years_limit: int | None = 3,
    ) -> list[dict[str, Any]]:
        # Step 1 – esearch: get PMIDs
        pmids = await asyncio.to_thread(self._esearch, query, max_results, years_limit)
        if not pmids:
            return []

        # Step 2 – efetch: get summaries
        summaries = await asyncio.to_thread(self._efetch_summaries, pmids)
        return summaries

    def _esearch(self, query: str, max_results: int, years_limit: int | None) -> list[str]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": str(min(max_results * 2, 20)),
            "retmode": "json",
            "sort": "relevance",
            "tool": self._TOOL,
            "email": self._EMAIL,
        }
        if years_limit:
            current_year = datetime.now().year
            params["mindate"] = str(current_year - years_limit)
            params["maxdate"] = str(current_year)
            params["datetype"] = "pdat"

        url = f"{_PUBMED_BASE}/esearch.fcgi?" + urllib.parse.urlencode(params)
        data = _http_get(url)
        if not isinstance(data, dict):
            return []
        return list(data.get("esearchresult", {}).get("idlist", []))

    def _efetch_summaries(self, pmids: list[str]) -> list[dict[str, Any]]:
        if not pmids:
            return []
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
            "rettype": "abstract",
            "tool": self._TOOL,
            "email": self._EMAIL,
        }
        url = f"{_PUBMED_BASE}/esummary.fcgi?" + urllib.parse.urlencode(params)
        data = _http_get(url)
        if not isinstance(data, dict):
            return []

        result_set = data.get("result", {})
        papers: list[dict[str, Any]] = []
        for pmid in result_set.get("uids", []):
            item = result_set.get(pmid, {})
            if not item:
                continue
            papers.append(self._normalise(pmid, item))
        return papers

    def _normalise(self, pmid: str, item: dict[str, Any]) -> dict[str, Any]:
        authors = [
            a.get("name", "") for a in (item.get("authors") or [])
        ]
        pub_date = item.get("pubdate") or item.get("sortpubdate") or ""
        year: int | None = None
        if pub_date:
            m = re.search(r"(\d{4})", pub_date)
            if m:
                year = int(m.group(1))

        title = item.get("title") or ""
        # PubMed titles sometimes have trailing period
        title = title.rstrip(".")

        doi = ""
        for id_info in item.get("articleids", []):
            if id_info.get("idtype") == "doi":
                doi = id_info.get("value", "")
                break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": item.get("description") or "",  # summary doesn't include abstract text
            "url": url,
            "doi": doi,
            "source": "pubmed",
            "arxiv_id": "",
            "pmid": pmid,
            "s2_id": "",
            "published": pub_date,
            "citations": 0,
        }


# ---------------------------------------------------------------------------
# Unified multi-source search
# ---------------------------------------------------------------------------

class PaperSearchClient:
    """
    Unified academic paper search across arXiv, Semantic Scholar, and PubMed.

    Strategy:
      - Run all enabled sources in parallel.
      - Deduplicate by title (normalised) and DOI.
      - Return results ranked by citation count (desc), then source priority.
    """

    _ARXIV_PRIORITY = 0
    _S2_PRIORITY = 1
    _PUBMED_PRIORITY = 2

    def __init__(
        self,
        use_arxiv: bool = True,
        use_semantic_scholar: bool = True,
        use_pubmed: bool = False,
    ) -> None:
        self._arxiv = ArxivClient() if use_arxiv else None
        self._s2 = SemanticScholarClient() if use_semantic_scholar else None
        self._pubmed = PubMedClient() if use_pubmed else None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        years_limit: int | None = 3,
        sort_by: str = "relevance",
        sources: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search across all enabled sources and return deduplicated results.

        Args:
            query:        Search query.
            max_results:  Total papers to return.
            years_limit:  Only include papers from last N years (None = no limit).
            sort_by:      "relevance" | "date" | "citations".
            sources:      Subset to use: ["arxiv", "semantic_scholar", "pubmed"].
                          None means use all configured clients.
        """
        query = (query or "").strip()
        if not query:
            return []

        fetch = min(max_results + 5, 20)  # fetch a bit more to survive dedup
        tasks: list[asyncio.Task[list[dict[str, Any]]]] = []

        active_sources: set[str] = set(sources) if sources else {"arxiv", "semantic_scholar", "pubmed"}

        if self._arxiv and "arxiv" in active_sources:
            tasks.append(asyncio.create_task(
                self._safe(self._arxiv.search(query, fetch, years_limit, sort_by), "arxiv")
            ))
        if self._s2 and "semantic_scholar" in active_sources:
            tasks.append(asyncio.create_task(
                self._safe(self._s2.search(query, fetch, years_limit), "semantic_scholar")
            ))
        if self._pubmed and "pubmed" in active_sources:
            tasks.append(asyncio.create_task(
                self._safe(self._pubmed.search(query, fetch, years_limit), "pubmed")
            ))

        if not tasks:
            return []

        gathered = await asyncio.gather(*tasks, return_exceptions=False)
        all_papers: list[dict[str, Any]] = []
        for batch in gathered:
            all_papers.extend(batch)

        deduped = self._deduplicate(all_papers)
        ranked = self._rank(deduped, sort_by)
        return ranked[:max_results]

    # ------------------------------------------------------------------
    async def _safe(self, coro: Any, source: str) -> list[dict[str, Any]]:
        try:
            result = await asyncio.wait_for(coro, timeout=_REQUEST_TIMEOUT_S + 5)
            return result or []
        except asyncio.TimeoutError:
            logger.warning("paper_search source '%s' timed out", source)
            return []
        except Exception as exc:
            logger.warning("paper_search source '%s' failed: %s", source, exc)
            return []

    def _normalise_title(self, title: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", title.lower())).strip()

    def _deduplicate(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate by DOI (exact) then by normalised title (fuzzy)."""
        seen_doi: set[str] = set()
        seen_title: set[str] = set()
        out: list[dict[str, Any]] = []
        for p in papers:
            doi = (p.get("doi") or "").strip().lower()
            norm_title = self._normalise_title(p.get("title") or "")
            if doi and doi in seen_doi:
                continue
            if norm_title and norm_title in seen_title:
                continue
            if doi:
                seen_doi.add(doi)
            if norm_title:
                seen_title.add(norm_title)
            out.append(p)
        return out

    def _rank(self, papers: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
        source_priority = {
            "semantic_scholar": 0,
            "arxiv": 1,
            "pubmed": 2,
        }
        if sort_by == "citations":
            return sorted(papers, key=lambda p: -(p.get("citations") or 0))
        if sort_by == "date":
            return sorted(papers, key=lambda p: (p.get("published") or ""), reverse=True)
        # Default: relevance — keep source-internal order but interleave by priority
        return sorted(papers, key=lambda p: source_priority.get(p.get("source", ""), 99))


# ---------------------------------------------------------------------------
# Formatting helpers (used by builtin wrapper)
# ---------------------------------------------------------------------------

def format_papers_as_text(papers: list[dict[str, Any]], max_abstract: int = 500) -> str:
    """Format papers list as readable Markdown text for LLM consumption."""
    if not papers:
        return "No papers found."
    lines: list[str] = []
    for p in papers:
        authors = p.get("authors") or []
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        year = p.get("year") or "?"
        source_tag = p.get("source", "").replace("_", " ").title()
        title = p.get("title") or "(Untitled)"
        url = p.get("url") or ""
        doi = p.get("doi") or ""
        arxiv_id = p.get("arxiv_id") or ""
        citations = int(p.get("citations") or 0)
        abstract = (p.get("abstract") or "")[:max_abstract]

        lines.append(f"**{title}** ({year}) [{source_tag}]")
        lines.append(f"Authors: {author_str or 'Unknown'}")
        if arxiv_id:
            lines.append(f"arXiv: {arxiv_id}")
        if doi:
            lines.append(f"DOI: {doi}")
        if url:
            lines.append(f"URL: {url}")
        if citations:
            lines.append(f"Citations: {citations}")
        if abstract:
            lines.append(f"Abstract: {abstract}")
        lines.append("")
    return "\n".join(lines)


def format_paper_citation(paper: dict[str, Any]) -> str:
    """Return a short inline citation string."""
    authors = paper.get("authors") or []
    year = paper.get("year") or "?"
    if not authors:
        return f"(Unknown, {year})"
    last_name = authors[0].split()[-1] if authors[0].split() else authors[0]
    if len(authors) > 1:
        return f"({last_name} et al., {year})"
    return f"({last_name}, {year})"


# ---------------------------------------------------------------------------
# Legacy-compat: ArxivSearchTool (used by existing imports)
# ---------------------------------------------------------------------------

class ArxivSearchTool:
    """Backward-compatible wrapper used by tutorbot and older code."""

    def __init__(self) -> None:
        self._client = ArxivClient()

    async def search_papers(
        self,
        query: str,
        max_results: int = 3,
        years_limit: int | None = 3,
        sort_by: str = "relevance",
    ) -> list[dict[str, Any]]:
        return await self._client.search(query, max_results, years_limit, sort_by)

    def format_paper_citation(self, paper: dict[str, Any]) -> str:
        return format_paper_citation(paper)

    def extract_arxiv_id_from_url(self, url: str) -> str | None:
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
        return match.group(1) if match else None


# Backward compatibility alias
PaperSearchTool = ArxivSearchTool


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_DEFAULT_CLIENT: PaperSearchClient | None = None


def get_paper_search_client() -> PaperSearchClient:
    """Return a singleton PaperSearchClient (arXiv + Semantic Scholar)."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = PaperSearchClient(
            use_arxiv=True,
            use_semantic_scholar=True,
            use_pubmed=False,  # off by default — biomedical only
        )
    return _DEFAULT_CLIENT


async def search_papers(
    query: str,
    max_results: int = 5,
    years_limit: int | None = 3,
    sort_by: str = "relevance",
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Module-level convenience function for multi-source paper search."""
    client = get_paper_search_client()
    return await client.search(
        query=query,
        max_results=max_results,
        years_limit=years_limit,
        sort_by=sort_by,
        sources=sources,
    )
