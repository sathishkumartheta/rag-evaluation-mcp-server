import gradio as gr
from retriever_eval_tools import (
    bm25_relevance_scorer,
    semantic_relevance_scorer,
    redundancy_checker,
    exact_match_checker,
)

from generator_eval_tools import (
    repetition_checker,
    semantic_diversity_checker,
    length_consistency_checker,
)

from system_eval_tools import (
    relevance_evaluator,
    coverage_evaluator,
    hallucination_detector,
)

# Retriever tools
bm25_tool = gr.Interface(
    fn=bm25_relevance_scorer,
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
)

semantic_tool = gr.Interface(
    fn=semantic_relevance_scorer,
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
)

redundancy_tool = gr.Interface(
    fn=redundancy_checker,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
)

exact_match_tool = gr.Interface(
    fn=exact_match_checker,
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
)

# Generator tools
repetition_tool = gr.Interface(
    fn=repetition_checker,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
)

semantic_diversity_tool = gr.Interface(
    fn=semantic_diversity_checker,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
)

length_consistency_tool = gr.Interface(
    fn=length_consistency_checker,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
)

# System tools
relevance_eval_tool = gr.Interface(
    fn=relevance_evaluator,
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
)

coverage_eval_tool = gr.Interface(
    fn=coverage_evaluator,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
)

hallucination_tool = gr.Interface(
    fn=hallucination_detector,
    inputs=[gr.Textbox(label="Generation (single)"), gr.Textbox(label="Source Documents")],
    outputs=gr.JSON(),
)

# Final tabbed UI
demo = gr.TabbedInterface(
    [
        hallucination_tool,
        relevance_eval_tool,
        coverage_eval_tool,
        bm25_tool,
        semantic_tool,
        redundancy_tool,
        exact_match_tool,
        repetition_tool,
        semantic_diversity_tool,
        length_consistency_tool
        
    ],
    [
        "RAG:System Hallucination",
        "RAG:System Relevance",
        "RAG:System Coverage",
        "Retriever:BM25 relevance",
        "Retriever:Semantic relevance",
        "Retriever: Redundancy",
        "Retriever:Exact Match",
        "Generator:Repetition",
        "Generator:Semantic Diversity",
        "Generator:Length Consistency"
        
    ]
)

demo.launch(mcp_server=True, share=True)
