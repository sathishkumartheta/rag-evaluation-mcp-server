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
