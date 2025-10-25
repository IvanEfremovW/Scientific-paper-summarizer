import gradio as gr
from pathlib import Path

from summarizer.summarizer import Summarizer
from summarizer.ingestion import exctract_text_from_document

def main(docuement_path: str | Path) -> str:
    summarizer = Summarizer()
   
    text = exctract_text_from_document(docuement_path)
    summary = summarizer.summarize(text)

    return summary
    

interface = gr.Interface(
    fn=main,
    inputs=gr.File(label="Upload a File"), 
    outputs=gr.Textbox(label="Summary", lines=25),
)

interface.launch()
