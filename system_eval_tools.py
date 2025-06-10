from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
import re
from retriever_eval_tools import parse_documents

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def relevance_score(query: str, answer: str) -> float:
    """
    Compute cosine similarity between the query and the answer.

    Args:
        query (str): The input query string.
        answer (str): The system-generated answer.

    Returns:
        float: Cosine similarity score between query and answer (0 to 1).
    """
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    answer_emb = embedding_model.encode(answer, convert_to_tensor=True)
    return round(util.cos_sim(query_emb, answer_emb).item(), 4)

def relevance_evaluator(query: str, generations: str) -> Dict:
    """
    Evaluate how relevant each generation is to the given query using cosine similarity.

    Args:
        query (str): The original input query.
        generations (str): Newline-separated or paragraph-separated list of generated responses.

    Returns:
        Dict: Relevance scores for each generation and average relevance.
    """
    generation_list = [g.strip() for g in generations.strip().splitlines() if g.strip()]
    if not generation_list:
        return {"error": "No valid generations provided."}

    scores = [relevance_score(query, gen) for gen in generation_list]
    avg_score = round(sum(scores) / len(scores), 4)

    results = [
        {"generation": gen, "relevance": score}
        for gen, score in zip(generation_list, scores)
    ]

    return {
        "tool": "System Relevance Evaluator",
        "query": query,
        "average_relevance": avg_score,
        "results": results
    }

def coverage_evaluator(_, generations: str) -> Dict:
    """
    Evaluate how diverse the content is across multiple system outputs (coverage proxy).

    Args:
        _ (str): Placeholder.
        generations (str): Newline-separated or paragraph-separated list of generated outputs.

    Returns:
        Dict: Pairwise cosine similarities and average to estimate content spread.
    """
    generation_list = [g.strip() for g in generations.strip().splitlines() if g.strip()]
    if len(generation_list) < 2:
        return {"error": "At least two generations required for coverage analysis."}

    emb = embedding_model.encode(generation_list, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb, emb)
    pairwise = []
    sim_sum = 0.0
    count = 0

    for i in range(len(generation_list)):
        for j in range(i + 1, len(generation_list)):
            score = sim_matrix[i][j].item()
            pairwise.append({
                "output_i": generation_list[i],
                "output_j": generation_list[j],
                "similarity": round(score, 4)
            })
            sim_sum += score
            count += 1

    avg_sim = round(sim_sum / count, 4)

    return {
        "tool": "System Coverage Evaluator",
        "average_pairwise_similarity": avg_sim,
        "pairwise_comparisons": pairwise
    }




def hallucination_detector(generation: str, source_docs: str) -> Dict:
    """
    Detects hallucinations by comparing generation sentences to source sentences using cosine similarity.
    Flags generation sentences with max similarity < 0.75 as hallucinated.

    Args:
        generation (str): The LLM-generated answer.
        source_docs (str): Supporting documents (raw string, newline/paragraph/JSON-style list).

    Returns:
        Dict: Hallucination flags and their similarity scores.
    """
    import re
    from sentence_transformers import SentenceTransformer, util
    from retriever_eval_tools import parse_documents

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Sentence splitting
    gen_sents = [s.strip() for s in re.split(r'[.?!]', generation) if s.strip()]
    doc_list = parse_documents(source_docs)
    doc_text = " ".join(doc_list)
    doc_sents = [s.strip() for s in re.split(r'[.?!]', doc_text) if s.strip()]

    if not gen_sents:
        return {"error": "No valid generation sentences."}
    if not doc_sents:
        return {"error": "No valid source sentences."}

    gen_embs = model.encode(gen_sents, convert_to_tensor=True)
    doc_embs = model.encode(doc_sents, convert_to_tensor=True)

    sim_matrix = util.cos_sim(gen_embs, doc_embs)
    threshold = 0.75
    flagged = []

    for i, gen_sent in enumerate(gen_sents):
        max_score = max(sim_matrix[i]).item()
        flagged.append({
            "sentence": gen_sent,
            "max_support_score": round(max_score, 4),
            "hallucinated": max_score < threshold
        })

    hallucinated_only = [f for f in flagged if f["hallucinated"]]
    return {
        "tool": "Hallucination Detector",
        "threshold": threshold,
        "results": hallucinated_only if hallucinated_only else "No hallucinated sentences detected.",
        "debug_scores": flagged  # include all scores for manual inspection
    }


