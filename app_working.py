# app.py

import gradio as gr
from retriever_eval_tools import bm25_relevance_scorer

# Top-level Gradio interface that directly connects UI to the tool function

gr.Interface(
    fn=bm25_relevance_scorer,
    inputs=[
        gr.Textbox(label="Query"),
        gr.Textbox(label="Retrieved Documents (one per line)", lines=6)
    ],
    outputs=gr.JSON(label="Relevance Scores")
).launch(mcp_server=True,share=True)
