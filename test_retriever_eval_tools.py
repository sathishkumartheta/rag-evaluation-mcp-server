# test_retriever_eval_tools.py

from retriever_eval_tools import bm25_relevance_scorer

def test_bm25_relevance_scorer():
    query = "how does the model scoring protocol work"
    documents = """The Model Scoring Protocol (MSP) defines how agents evaluate and compare tool outputs.
MSP is used to rank multiple completions from different tools based on utility.
Model Serving Platform (MSP) manages cloud deployment and versioning.
The protocol includes a scoring rubric for structured and free-form outputs.
This document describes HTTP request routing in a microservice architecture."""

    result = bm25_relevance_scorer(query, documents)

    assert "results" in result, "Output should contain 'results'"
    assert len(result["results"]) == 5, "Should return a score for each document"

    print("✅ Test Passed. Output:")
    for i, item in enumerate(result["results"], start=1):
        print(f"\nDoc {i}:\nText: {item['document']}\nBM25 Score: {item['score']}")

if __name__ == "__main__":
    test_bm25_relevance_scorer()



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


########################################

from typing import Dict
from rank_bm25 import BM25Okapi
import re
import ast

def simple_tokenize(text: str):
    # Lowercase + remove punctuation; returns list of words
    return re.findall(r"\b\w+\b", text.lower())

def parse_documents(documents: str):
    """
    Parse the input documents string into a clean list of non-empty documents.
    Supports:
      - Newline-separated
      - Python-style list (e.g., "['doc1', 'doc2']")
    """
    doc_list = []

    # Try to parse as Python list if string looks like it
    if documents.strip().startswith("[") and documents.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(documents)
            if isinstance(parsed, list):
                doc_list = [d.strip() for d in parsed if isinstance(d, str) and d.strip()]
        except:
            pass  # Fall back to line split

    if not doc_list:
        # Default to newline splitting
        doc_list = [d.strip() for d in documents.splitlines() if d.strip()]

    return doc_list

def bm25_relevance_scorer(query: str, documents: str) -> Dict:
    """
    Computes lexical BM25 relevance scores between a query and multiple documents.

    Args:
        query (str): The input search query.
        documents (str): Newline-separated or list-style string of documents.

    Returns:
        Dict: A dictionary containing:
            - the original query
            - results: list of {document, score}
    """
    doc_list = parse_documents(documents)
    if not query.strip() or not doc_list:
        return {"error": "Query and documents must be non-empty."}

    tokenized_docs = [simple_tokenize(doc) for doc in doc_list]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = simple_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    results = []
    for doc, score in zip(doc_list, bm25_scores):
        safe_score = max(0.0, round(float(score), 4))  # Clamp to avoid negatives
        results.append({
            "document": doc,
            "score": safe_score
        })

    return {
        "tool": "BM25 Relevance Scorer",
        "query": query,
        "results": results
    }
