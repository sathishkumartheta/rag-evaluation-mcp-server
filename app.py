import gradio as gr
from retriever_eval_tools import (
    bm25_relevance_scorer,
    semantic_relevance_scorer,
    redundancy_checker,
    exact_match_checker,
    ranking_consistency_checker,
    document_novelty_scorer,
    topic_diversity_evaluator,
)

from generator_eval_tools import (
    repetition_checker,
    semantic_diversity_checker,
    length_consistency_checker,
)

from system_eval_tools import (
    relevance_evaluator,
    coverage_evaluator,
)

with gr.Blocks() as demo:
    with gr.Tab("Retriever Eval"):
        gr.Markdown("## Retriever Evaluation Tools")
        gr.Interface(bm25_relevance_scorer, [gr.Textbox(label="Query"), gr.Textbox(label="Documents")], gr.JSON()).render()
        gr.Interface(semantic_relevance_scorer, [gr.Textbox(label="Query"), gr.Textbox(label="Documents")], gr.JSON()).render()
        gr.Interface(redundancy_checker, [gr.Textbox(label="Unused"), gr.Textbox(label="Documents")], gr.JSON()).render()
        gr.Interface(exact_match_checker, [gr.Textbox(label="Query"), gr.Textbox(label="Documents")], gr.JSON()).render()
        gr.Interface(ranking_consistency_checker, [
            gr.Textbox(label="Original Query"),
            gr.Textbox(label="Paraphrased Query"),
            gr.Textbox(label="Documents")
        ], gr.JSON()).render()
        gr.Interface(document_novelty_scorer, [gr.Textbox(label="Documents")], gr.JSON()).render()
        gr.Interface(topic_diversity_evaluator, [
            gr.Textbox(label="Documents"),
            gr.Number(label="Number of Clusters", value=3)
        ], gr.JSON()).render()

    with gr.Tab("Generator Eval"):
        gr.Markdown("## Generator Evaluation Tools")
        gr.Interface(repetition_checker, [gr.Textbox(label="Unused"), gr.Textbox(label="Generations")], gr.JSON()).render()
        gr.Interface(semantic_diversity_checker, [gr.Textbox(label="Unused"), gr.Textbox(label="Generations")], gr.JSON()).render()
        gr.Interface(length_consistency_checker, [gr.Textbox(label="Unused"), gr.Textbox(label="Generations")], gr.JSON()).render()

    with gr.Tab("System Eval"):
        gr.Markdown("## System Evaluation Tools")
        gr.Interface(relevance_evaluator, [gr.Textbox(label="Query"), gr.Textbox(label="Generations")], gr.JSON()).render()
        gr.Interface(coverage_evaluator, [gr.Textbox(label="Unused"), gr.Textbox(label="Generations")], gr.JSON()).render()

demo.launch(share=True, mcp_server=True)
