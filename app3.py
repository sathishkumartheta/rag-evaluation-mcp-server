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
    examples=[["What is the capital of France?", "1. Paris is the capital of France.\n2. Berlin is in Germany.\n3. Madrid is in Spain."]]
)

semantic_tool = gr.Interface(
    fn=semantic_relevance_scorer,
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
    examples=[["What causes rain?", "1. Rain is caused by condensation of water vapor.\n2. The Earth revolves around the sun.\n3. Water evaporates and returns as rain."]]
)

redundancy_tool = gr.Interface(
    fn=redundancy_checker,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
    examples=[["_", "1. Apples are red.\n2. Apples are red and juicy.\n3. Oranges are orange in color."]]
)

exact_match_tool = gr.Interface(
    fn=exact_match_checker,
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Documents")],
    outputs=gr.JSON(),
    examples=[["capital of France", "1. Paris is the capital of France.\n2. Berlin is in Germany.\n3. The Eiffel Tower is in Paris."]]
)

# Generator tools
repetition_tool = gr.Interface(
    fn=repetition_checker,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
    examples=[["_", "The cat is on the mat. The cat is on the mat.\nDogs bark loudly."]]
)

semantic_diversity_tool = gr.Interface(
    fn=semantic_diversity_checker,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
    examples=[["_", "1. The sky is blue.\n2. It is sunny today.\n3. The sky is blue."]]
)

length_consistency_tool = gr.Interface(
    fn=length_consistency_checker,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
    examples=[["_", "1. The dog barks.\n2. Cats are quiet and sleep often.\n3. Birds sing."]]
)

# System tools
relevance_eval_tool = gr.Interface(
    fn=relevance_evaluator,
    inputs=[gr.Textbox(label="Query"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
    examples=[["What are the benefits of exercise?", "1. Exercise improves cardiovascular health.\n2. Eating vegetables is healthy."]]
)

coverage_eval_tool = gr.Interface(
    fn=coverage_evaluator,
    inputs=[gr.Textbox(label="Unused"), gr.Textbox(label="Generations")],
    outputs=gr.JSON(),
    examples=[["_", "1. Apples are good for health.\n2. Apples can be red or green.\n3. Eating apples helps digestion."]]
)

hallucination_tool = gr.Interface(
    fn=hallucination_detector,
    inputs=[gr.Textbox(label="Generation (single)"), gr.Textbox(label="Source Documents")],
    outputs=gr.JSON(),
    examples=[[
        "Albert Einstein invented the light bulb.  Albert Einstein developed the theory of relativity.",
        "Albert Einstein was a theoretical physicist known for the theory of relativity.\n The light bulb was invented by Thomas Edison in the late 19th century.\n Albert Einstein received the Nobel Prize in Physics in 1921 for his work on the photoelectric effect."
        
    ]]
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