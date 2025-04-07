import gradio as gr
from summerizer import summarize
gr.Interface(fn=summarize, inputs="textbox", outputs="textbox", title="ArXiv Paper Summarizer").launch()
