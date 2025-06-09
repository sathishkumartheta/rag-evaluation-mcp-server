import gradio as gr
from retriever_eval_tools import (
    bm25_relevance_scorer,
    semantic_relevance_scorer,
    redundancy_checker,
    exact_match_checker,
)

# Each tool defined without rendering now
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
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
)

exact_match_tool = gr.Interface(
    fn=exact_match_checker,
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
)

# Only this gets rendered
demo = gr.TabbedInterface(
    [bm25_tool, semantic_tool, redundancy_tool, exact_match_tool],
    ["BM25", "Semantic", "Redundancy", "Exact Match"]
)

demo.launch(mcp_server=True, share=True)
