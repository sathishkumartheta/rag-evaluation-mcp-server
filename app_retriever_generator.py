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

# Tabbed UI with all tools
demo = gr.TabbedInterface(
    [
        bm25_tool,
        semantic_tool,
        redundancy_tool,
        exact_match_tool,
        repetition_tool,
        semantic_diversity_tool,
        length_consistency_tool
    ],
    [
        "BM25",
        "Semantic",
        "Redundancy",
        "Exact Match",
        "Repetition",
        "Semantic Diversity",
        "Length Consistency"
    ]
)

demo.launch(mcp_server=True, share=True)
