"""
Pre-Retrieval Module for Modular RAG
"""

from typing import List, Dict, Tuple, Optional, Any
from src.lego_rag.logger import logger
import uuid
import re
import json

from langchain_core.language_models.chat_models import BaseChatModel

# PreRetrievalPipeline


# ============================================================
# Utility
# ============================================================

def call_llm(llm: BaseChatModel, prompt: str) -> str:
    """
    Canonical way to call a LangChain chat model.
    """
    return llm.invoke(prompt).content.strip()


# ============================================================
# Query Expansion
# ============================================================

class QueryExpander:
    def __init__(
        self,
        llm: BaseChatModel,
        max_expansions: int = 3,
        weight_original: float = 0.6,
    ):
        self.llm = llm
        self.max_expansions = max_expansions
        self.weight_original = float(weight_original)

    def multi_query(self, q: str) -> List[Tuple[str, float]]:
        prompt = (
            f"Generate {self.max_expansions} concise paraphrases or focused variants "
            f"of the following query. Return a JSON array of strings.\n\n"
            f"Query: {json.dumps(q)}"
        )

        response = call_llm(self.llm, prompt)
        variants = self._safe_parse_json_array(response)

        out = [(q, self.weight_original)]
        seen = {self._norm(q)}

        for v in variants:
            if self._norm(v) in seen:
                continue
            out.append((v, (1.0 - self.weight_original) / self.max_expansions))
            seen.add(self._norm(v))

        return out

    def subquery_decompose(self, q: str) -> List[str]:
        prompt = (
            "Decompose the following complex question into simpler sub-questions. "
            "Return a JSON array.\n\n"
            + json.dumps(q)
        )
        response = call_llm(self.llm, prompt)
        return self._safe_parse_json_array(response)

    def chain_of_verification(self, queries: List[str]) -> List[Tuple[str, float]]:
        scored = []

        for q in queries:
            prompt = (
                "Rate how faithful and useful the following query is to the original intent "
                "on a scale from 0 to 1. Return only a number.\n\n"
                + json.dumps(q)
            )
            response = call_llm(self.llm, prompt)
            score = self._safe_parse_score(response)
            scored.append((q, score))

        filtered = [(q, s) for q, s in scored if s > 0.2] or scored
        total = sum(s for _, s in filtered) or 1.0

        return [(q, s / total) for q, s in filtered]

    # ---------------- helpers ----------------

    def _safe_parse_json_array(self, text: str) -> List[str]:
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass

        return [ln.strip("-• ") for ln in text.splitlines() if ln.strip()]

    def _safe_parse_score(self, text: str) -> float:
        try:
            num = float(re.findall(r"[0-9]*\.?[0-9]+", text)[0])
            return max(0.0, min(1.0, num))
        except Exception:
            return 0.0

    def _norm(self, t: str) -> str:
        return re.sub(r"\s+", " ", t.lower().strip())


# ============================================================
# Query Transformation
# ============================================================

class QueryTransformer:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def rewrite(self, q: str, domain_hint: str = "") -> str:
        prompt = (
            "Rewrite the following query to maximize retrieval quality. "
            "Keep it concise.\n\n"
            f"Query: {json.dumps(q)}\nContext: {json.dumps(domain_hint)}"
        )
        return call_llm(self.llm, prompt)

    def hyde(self, q: str) -> str:
        prompt = (
            "Write a concise hypothetical answer (1–3 sentences) that a relevant document "
            "might contain for the following question.\n\n"
            + json.dumps(q)
        )
        return call_llm(self.llm, prompt)

    def step_back(self, q: str) -> str:
        prompt = (
            "Write one high-level conceptual question that captures the broader idea "
            "behind the following query.\n\n"
            + json.dumps(q)
        )
        return call_llm(self.llm, prompt)


# ============================================================
# Query Construction (SQL / Cypher)
# ============================================================

class QueryConstructor:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm

    def to_sql(
        self,
        q: str,
        table_schemas: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        if self.llm:
            prompt = (
                "Translate the following request into a SAFE SQL SELECT query. "
                "Use the provided schema. Return only SQL.\n\n"
                f"Schema: {json.dumps(table_schemas or {})}\n\n"
                f"Request: {json.dumps(q)}"
            )
            return call_llm(self.llm, prompt)

        # deterministic fallback
        table = next(iter(table_schemas or {"documents": ["text"]}))
        column = (table_schemas or {"documents": ["text"]})[table][0]
        esc = q.replace("'", "''")

        return (
            f"SELECT * FROM {table} "
            f"WHERE {column} ILIKE '%{esc}%' "
            f"LIMIT 100;"
        )


# ============================================================
# Pipeline Orchestrator
# ============================================================

class PreRetrievalPipeline:
    def __init__(self, llm: BaseChatModel):
        self.expander = QueryExpander(llm)
        self.transformer = QueryTransformer(llm)
        self.constructor = QueryConstructor(llm)

    def run(
        self,
        q: str,
        do_hyde: bool = True,
        to_sql: bool = False,
        schema: Optional[Dict] = None,
    ) -> Dict:

        multi = self.expander.multi_query(q)
        subs = self.expander.subquery_decompose(q)

        combined = [t for t, _ in multi] + subs
        verified = self.expander.chain_of_verification(combined)

        candidates = []
        for text, weight in verified:
            rewritten = self.transformer.rewrite(text)
            candidates.append({
                "id": str(uuid.uuid4()),
                "original": text,
                "text": rewritten,
                "weight": weight,
                "type": "expanded",
            })

        hyde_text = None
        if do_hyde:
            hyde_text = self.transformer.hyde(q)
            candidates.insert(0, {
                "id": str(uuid.uuid4()),
                "original": q,
                "text": hyde_text,
                "weight": 0.4,
                "type": "hyde",
            })

        total = sum(c["weight"] for c in candidates) or 1.0
        for c in candidates:
            c["weight"] /= total

        sql = self.constructor.to_sql(q, schema) if to_sql else None

        return {
            "candidates": candidates,
            "hyde": hyde_text,
            "sql": sql,
        }
    
#quick debugging
if __name__ == "__main__":
    #pass

    from pprint import pprint
    from langchain_groq import ChatGroq
    from settings import settings

    # -----------------------------
    # 1. Initialize a real Chat Model
    # -----------------------------

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
        api_key= settings.GROQ_API_KEY 
    )

    # -----------------------------
    # 2. Create pipeline
    # -----------------------------
    pipeline = PreRetrievalPipeline(llm=llm)

    # -----------------------------
    # 3. Test query
    # -----------------------------
    query = (
        "How does small-to-big retrieval work in RAG systems, "
        "and why is it better than fixed-size chunking?"
    )

    # -----------------------------
    # 4. Run pipeline
    # -----------------------------
    result = pipeline.run(
        q=query,
        do_hyde=True,
        to_sql=False,
    )

    # -----------------------------
    # 5. Pretty-print results
    # -----------------------------
    print("\n=== PRE-RETRIEVAL DEBUG OUTPUT ===\n")
    pprint(result)

    print("\n=== CANDIDATES (ORDERED) ===\n")
    for c in result["candidates"]:
        print(f"[{c['type']}] weight={c['weight']:.3f}")
        print(c["text"])
        print("-" * 80)