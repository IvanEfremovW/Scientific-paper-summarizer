import gradio as gr
from pathlib import Path

from .summarizer import Summarizer
from .ingestion import exctract_text_from_document


def pipeline(docuement_path: str | Path) -> str:
    summarizer = Summarizer()

    text = exctract_text_from_document(docuement_path)
    summary = summarizer.summarize(text)

    return summary


def main():
    interface = gr.Interface(
        fn=pipeline,
        inputs=gr.File(label="Upload a File"),
        outputs=gr.Textbox(label="Summary", lines=25),
    )

    interface.launch()


if __name__ == "__main__":
    main()
