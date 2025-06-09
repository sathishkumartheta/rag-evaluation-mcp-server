from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

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
