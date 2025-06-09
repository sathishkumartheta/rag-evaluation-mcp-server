from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
import re

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def simple_tokenize(text: str):
    """
    Tokenize text into lowercase word tokens using regex.

    Args:
        text (str): Input string.

    Returns:
        List[str]: List of tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())

def parse_outputs(outputs: str) -> List[str]:
    """
    Parse the generated outputs from a string format to a list of strings.

    Args:
        outputs (str): Raw string containing outputs, separated by newlines or paragraphs.

    Returns:
        List[str]: List of cleaned output strings.
    """
    lines = [line.strip() for line in outputs.strip().splitlines() if line.strip()]
    return lines if lines else [outputs.strip()] if outputs.strip() else []

def repetition_checker(_, generations: str) -> Dict:
    """
    Detects repetitive phrases or n-grams in generated text outputs.

    Args:
        _ (str): Placeholder for tool compatibility.
        generations (str): A string of generated outputs, separated by newlines or paragraphs.

    Returns:
        Dict: A report on detected repetitions per generation.
    """
    output_list = parse_outputs(generations)
    repetition_report = []

    for output in output_list:
        tokens = simple_tokenize(output)
        repeated = set()
        for i in range(len(tokens) - 2):
            trigram = " ".join(tokens[i:i+3])
            if tokens.count(trigram) > 1:
                repeated.add(trigram)
        repetition_report.append({
            "output": output,
            "repeated_phrases": list(repeated) if repeated else "None"
        })

    return {
        "tool": "Repetition Checker",
        "results": repetition_report
    }

def semantic_diversity_checker(_, generations: str) -> Dict:
    """
    Measures how semantically diverse the generated outputs are using cosine similarity.

    Args:
        _ (str): Placeholder for tool compatibility.
        generations (str): A string of generated outputs, separated by newlines or paragraphs.

    Returns:
        Dict: A report showing pairwise similarities and average similarity.
    """
    output_list = parse_outputs(generations)
    if len(output_list) < 2:
        return {"error": "At least two generations are needed to measure diversity."}

    emb = embedding_model.encode(output_list, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb, emb)
    pairwise = []
    sim_sum = 0.0
    count = 0

    for i in range(len(output_list)):
        for j in range(i + 1, len(output_list)):
            score = sim_matrix[i][j].item()
            sim_sum += score
            count += 1
            pairwise.append({
                "output_i": output_list[i],
                "output_j": output_list[j],
                "similarity": round(score, 4)
            })

    avg_sim = round(sim_sum / count, 4) if count else 0.0

    return {
        "tool": "Semantic Diversity Checker",
        "average_similarity": avg_sim,
        "pairwise_scores": pairwise
    }

def length_consistency_checker(_, generations: str) -> Dict:
    """
    Evaluates the consistency of lengths across multiple generated outputs.

    Args:
        _ (str): Placeholder for tool compatibility.
        generations (str): A string of generated outputs, separated by newlines or paragraphs.

    Returns:
        Dict: Statistics on lengths and list of outlier generations.
    """
    output_list = parse_outputs(generations)
    lengths = [len(output.split()) for output in output_list]
    avg_len = sum(lengths) / len(lengths)
    std_dev = (sum((l - avg_len) ** 2 for l in lengths) / len(lengths)) ** 0.5

    outliers = []
    for output, length in zip(output_list, lengths):
        if abs(length - avg_len) > 2 * std_dev:
            outliers.append({"output": output, "length": length})

    return {
        "tool": "Length Consistency Checker",
        "average_length": round(avg_len, 2),
        "std_deviation": round(std_dev, 2),
        "outliers": outliers
    }
