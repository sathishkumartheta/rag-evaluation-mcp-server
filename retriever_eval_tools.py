from typing import Dict,List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import re
import ast

# Load once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def simple_tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())

def parse_documents(documents: str):
    doc_list = []

    if documents.strip().startswith("[") and documents.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(documents)
            if isinstance(parsed, list):
                doc_list = [d.strip() for d in parsed if isinstance(d, str) and d.strip()]
                return doc_list
        except:
            pass

    matches = re.findall(r"\d+[.)]\s*(.+)", documents.strip())
    if matches:
        return [m.strip() for m in matches if m.strip()]

    lines = [line.strip() for line in documents.strip().splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines

    paras = [p.strip() for p in documents.split("\n\n") if p.strip()]
    if paras:
        return paras

    return [documents.strip()] if documents.strip() else []

# 1. BM25 Scorer 
def bm25_relevance_scorer(query: str, documents: str) -> Dict:
    """
    Compute relevance scores between a query and a list of documents using the BM25 algorithm.

    This tool tokenizes each document and the query using a simple whitespace and punctuation-based tokenizer,
    then calculates BM25 scores to measure how relevant each document is to the query.

    Args:
        query (str): The input search query in plain text.
        documents (str): A set of documents in raw string format. Supports JSON-style lists, paragraph-separated, or newline-separated entries.

    Returns:
        Dict: A dictionary containing:
            - 'tool': The name of the tool ("BM25 Relevance Scorer").
            - 'query': The input query.
            - 'results': A list of dictionaries, each with:
                - 'document': The original document string.
                - 'score': The BM25 relevance score (higher means more relevant).
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
        results.append({"document": doc, "score": round(max(score, 0.0), 4)})

    return {"tool": "BM25 Relevance Scorer", "query": query, "results": results}


# 2. Semantic Relevance (Cosine Similarity)
def semantic_relevance_scorer(query: str, documents: str) -> Dict:
    """
    Compute semantic relevance scores between a query and a list of documents using cosine similarity.

    This tool encodes the query and documents using a sentence embedding model, then calculates pairwise
    cosine similarity scores to determine how semantically similar each document is to the query.

    Args:
        query (str): The input query in natural language.
        documents (str): A string representing a list of documents. Supports multiple formats including JSON-style lists,
                         paragraph-separated text, or newline-separated entries.

    Returns:
        Dict: A dictionary containing:
            - 'tool': The name of the tool ("Semantic Relevance Scorer").
            - 'query': The input query.
            - 'results': A list of dictionaries with:
                - 'document': The original document text.
                - 'score': A float representing cosine similarity between the query and the document (0 to 1).
    """
    doc_list = parse_documents(documents)
    if not query.strip() or not doc_list:
        return {"error": "Query and documents must be non-empty."}

    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    doc_embs = embedding_model.encode(doc_list, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_emb, doc_embs)[0]

    results = []
    for doc, score in zip(doc_list, cosine_scores):
        results.append({"document": doc, "score": round(score.item(), 4)})

    return {"tool": "Semantic Relevance Scorer", "query": query, "results": results}


# 3. Redundancy Checker 
def redundancy_checker(_, documents: str) -> Dict:
    """
    Detect redundant or highly similar document pairs using semantic similarity.

    This tool encodes all documents into embeddings and computes cosine similarity scores between every pair.
    It flags document pairs with similarity greater than 0.8 as redundant.

    Args:
        _ (str): Placeholder for unused input (for LLM compatibility).
        documents (str): A string containing multiple documents. Accepts formats like JSON-style lists, 
                         newline-separated text, or paragraph-separated entries.

    Returns:
        Dict: A dictionary containing:
            - 'tool': The name of the tool ("Redundancy Checker").
            - 'results': A list of redundant document pairs with their similarity scores,
                         or a message indicating no redundancy if none are found.
    """
    doc_list = parse_documents(documents)
    if not doc_list or len(doc_list) < 2:
        return {"error": "At least two documents are required to check redundancy."}

    doc_embs = embedding_model.encode(doc_list, convert_to_tensor=True)
    sim_matrix = util.cos_sim(doc_embs, doc_embs)

    redundant_pairs = []
    for i in range(len(doc_list)):
        for j in range(i + 1, len(doc_list)):
            score = sim_matrix[i][j].item()
            if score > 0.8:
                redundant_pairs.append({
                    "doc_i": doc_list[i],
                    "doc_j": doc_list[j],
                    "similarity": round(score, 4)
                })

    return {
        "tool": "Redundancy Checker",
        "results": redundant_pairs if redundant_pairs else "No highly redundant documents found."
    }


# 4. Exact Match Checker 
def exact_match_checker(query: str, documents: str) -> Dict:
    """
    Check if each document contains the exact query string as a substring (case-insensitive).

    This tool performs a case-insensitive substring search to determine whether the query appears
    in each document exactly as entered (without tokenization or semantic matching).

    Args:
        query (str): The exact phrase or keyword to search for.
        documents (str): A raw string containing multiple documents. Supports JSON-style lists,
                         newline-separated, or paragraph-separated formats.

    Returns:
        Dict: A dictionary containing:
            - 'tool': The name of the tool ("Exact Match Checker").
            - 'query': The input query string.
            - 'results': A list of dictionaries with:
                - 'document': The original document.
                - 'exact_match': A boolean indicating if the query is an exact substring of the document.
    """
    doc_list = parse_documents(documents)
    query_lower = query.strip().lower()
    results = []

    for doc in doc_list:
        match = query_lower in doc.lower()
        results.append({
            "document": doc,
            "exact_match": match
        })

    return {
        "tool": "Exact Match Checker",
        "query": query,
        "results": results
    }

