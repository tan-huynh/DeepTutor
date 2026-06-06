from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deeptutor.tools.paper_search_tool import (
    ArxivClient,
    PaperSearchClient,
    PubMedClient,
    SemanticScholarClient,
    format_paper_citation,
    format_papers_as_text,
    get_paper_search_client,
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

ARXIV_MOCK_PAPER = {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "year": 2017,
    "abstract": "The dominant sequence transduction models...",
    "url": "http://arxiv.org/abs/1706.03762v5",
    "doi": "",
    "source": "arxiv",
    "arxiv_id": "1706.03762",
    "pmid": "",
    "s2_id": "",
    "published": "2017-06-12T17:57:34Z",
    "citations": 0,
}

S2_MOCK_PAPER = {
    "title": "Attention is All you Need",
    "authors": ["A. Vaswani", "N. Shazeer"],
    "year": 2017,
    "abstract": "The dominant sequence transduction models are based on...",
    "url": "https://www.semanticscholar.org/paper/123",
    "doi": "10.5555/3295222.3295349",
    "source": "semantic_scholar",
    "arxiv_id": "1706.03762",
    "pmid": "",
    "s2_id": "123",
    "published": "2017-12-04",
    "citations": 90000,
}

PUBMED_MOCK_PAPER = {
    "title": "A review of attention mechanisms in medical imaging.",
    "authors": ["Smith J", "Doe A"],
    "year": 2021,
    "abstract": "Attention mechanisms have shown great promise...",
    "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    "doi": "10.1016/j.media.2021.102000",
    "source": "pubmed",
    "arxiv_id": "",
    "pmid": "12345678",
    "s2_id": "",
    "published": "2021 Jan",
    "citations": 0,
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paper_search_client_deduplication(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set up mock clients that return slightly different versions of the same paper
    class MockArxivClient:
        async def search(self, *args, **kwargs) -> list[dict[str, Any]]:
            return [ARXIV_MOCK_PAPER.copy()]

    class MockS2Client:
        async def search(self, *args, **kwargs) -> list[dict[str, Any]]:
            return [S2_MOCK_PAPER.copy()]

    class MockPubMedClient:
        async def search(self, *args, **kwargs) -> list[dict[str, Any]]:
            return [PUBMED_MOCK_PAPER.copy()]

    client = PaperSearchClient()
    monkeypatch.setattr(client, "_arxiv", MockArxivClient())
    monkeypatch.setattr(client, "_s2", MockS2Client())
    monkeypatch.setattr(client, "_pubmed", MockPubMedClient())

    # Search with deduplication enabled (default behavior)
    results = await client.search("attention is all you need", max_results=5, sources=["arxiv", "semantic_scholar", "pubmed"])
    
    # Arxiv and S2 papers should be deduplicated (they share title similarity or arxiv_id usually, but our logic checks doi and norm_title)
    # The normalise_title logic: "Attention Is All You Need" -> "attention is all you need"
    # So S2 and Arxiv will match on title. PubMed is different.
    assert len(results) == 2
    
    # Because S2 has citation count 90000 and default sort is 'relevance', let's check what sort order does
    # Priority is: s2 (0), arxiv (1), pubmed (2)
    sources_in_results = [r["source"] for r in results]
    assert "pubmed" in sources_in_results
    assert ("semantic_scholar" in sources_in_results or "arxiv" in sources_in_results)

@pytest.mark.asyncio
async def test_paper_search_client_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    client = PaperSearchClient()
    papers = [
        ARXIV_MOCK_PAPER.copy(),
        S2_MOCK_PAPER.copy(),
        PUBMED_MOCK_PAPER.copy()
    ]
    
    # Test sort by citations
    ranked_by_citations = client._rank(papers, sort_by="citations")
    assert ranked_by_citations[0]["source"] == "semantic_scholar"  # S2 has 90000
    
    # Test sort by date
    # Arxiv: 2017-06-12T17:57:34Z
    # S2: 2017-12-04
    # PubMed: 2021 Jan
    ranked_by_date = client._rank(papers, sort_by="date")
    assert ranked_by_date[0]["source"] == "pubmed" # 2021 is newest
    
    # Test sort by relevance (source priority)
    ranked_by_relevance = client._rank(papers, sort_by="relevance")
    assert ranked_by_relevance[0]["source"] == "semantic_scholar" # Priority 0
    assert ranked_by_relevance[1]["source"] == "arxiv" # Priority 1
    assert ranked_by_relevance[2]["source"] == "pubmed" # Priority 2

def test_format_paper_citation() -> None:
    # Multiple authors
    citation1 = format_paper_citation(ARXIV_MOCK_PAPER)
    assert citation1 == "(Vaswani et al., 2017)"
    
    # Single author
    paper2 = ARXIV_MOCK_PAPER.copy()
    paper2["authors"] = ["Alan Turing"]
    citation2 = format_paper_citation(paper2)
    assert citation2 == "(Turing, 2017)"
    
    # No authors
    paper3 = ARXIV_MOCK_PAPER.copy()
    paper3["authors"] = []
    citation3 = format_paper_citation(paper3)
    assert citation3 == "(Unknown, 2017)"

def test_format_papers_as_text() -> None:
    text = format_papers_as_text([S2_MOCK_PAPER, PUBMED_MOCK_PAPER])
    assert "Attention is All you Need" in text
    assert "[Semantic Scholar]" in text
    assert "Citations: 90000" in text
    assert "[Pubmed]" in text
    assert "URL: https://pubmed.ncbi.nlm.nih.gov/12345678/" in text

@pytest.mark.asyncio
async def test_get_paper_search_client_singleton() -> None:
    client1 = get_paper_search_client()
    client2 = get_paper_search_client()
    assert client1 is client2
