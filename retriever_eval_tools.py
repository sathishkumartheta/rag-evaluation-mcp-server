# retriever_evaluation_tools.py

from typing import Dict
from rank_bm25 import BM25Okapi


def bm25_relevance_scorer(query: str, documents: str) -> Dict:
    """
    Computes lexical relevance scores between a query and a list of newline-separated documents using BM25.

    This tool measures token-level overlap and importance to score how relevant each document is
    to the input query. It's particularly effective for matching exact or near-exact phrases.

    Args:
        query (str): The input search query.
        documents (str): Newline-separated string of retrieved documents.

    Returns:
        Dict: A dictionary with:
              - the original query,
              - and a list of documents each with:
                  - document text,
                  - BM25 score.
    """
    doc_list = [d.strip() for d in documents.splitlines() if d.strip()]
    if not query or not doc_list:
        return {"error": "Query and documents must be non-empty strings."}

    tokenized_docs = [doc.split() for doc in doc_list]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.split())

    results = []
    for doc, score in zip(doc_list, bm25_scores):
        results.append({
            "document": doc,
            "score": round(float(score), 4)
        })

    return {
        "tool": "BM25 Relevance Scorer",
        "query": query,
        "results": results
    }
