from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import BitsAndBytesConfig
import torch


class Summarizer:
    def __init__(self, chunk_size: int = 10000, chunk_overlap: int = 200) -> None:
        self.chat_model = self._setup_llm_model()

        self.MAP_PROMPT = PromptTemplate.from_template(
            """
            You are an expert summarizer. Perform Chain-of-Density (CoD) summarization on the following text in 4 iterative steps:
            
            1. Start with a concise summary (1–2 sentences).
            2. In each subsequent iteration:
            - Add new salient entities (people, organizations, locations, dates, events, technical terms, or key concepts) that were missing.
            - Preserve the summary length (~3–4 sentences).
            - Increase information density without sacrificing clarity.
            3. Return ONLY the final summary after the last iteration. Do not include intermediate steps, explanations, or markdown.

            Text:
            {text}
            """
        )
        self.map_chain = self.MAP_PROMPT | self.chat_model | StrOutputParser()

        self.REDUCE_PROMPT = PromptTemplate.from_template(
            """
            You are an expert summarizer. Create a unified, high-density summary of the entire document by performing Chain-of-Density (CoD) summarization on the following partial summaries in 4 iterative steps:
            
            1. Begin with a coherent synthesis of all input summaries (2–3 sentences).
            2. In each subsequent iteration:
            - Integrate additional distinct entities and key facts from different parts of the input.
            - Resolve redundancies and ensure logical flow.
            - Maintain a consistent length (~4–5 sentences).
            - Maximize information density while preserving accuracy.
            3. Return ONLY the final, polished summary in 4-5 sentences. Do not include iteration steps, commentary, or formatting.
            
            Partial summaries:
            {text}
            """
        )
        self.reduce_chain = self.REDUCE_PROMPT | self.chat_model | StrOutputParser()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _setup_llm_model(self):
        from summarizer.config import SUMMARIZATION_MODEL, USE_4BIT

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if USE_4BIT
            else None
        )

        llm = HuggingFacePipeline.from_model_id(
            model_id=SUMMARIZATION_MODEL,
            task="text-generation",
            device_map="auto",
            model_kwargs=dict(quantization_config=quantization_config),
            pipeline_kwargs=dict(
                max_new_tokens=512,
                repetition_penalty=1.03,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
            ),
        )

        chat_model = ChatHuggingFace(llm=llm)

        return chat_model

    def summarize(self, text: str) -> str:
        if not text.strip():
            return ""

        chunks = self.text_splitter.split_text(text)

        mapped_summaries = [self.map_chain.invoke({"text": chunk}) for chunk in chunks]
        reduced_summary = self.reduce_chain.invoke({"text": "".join(mapped_summaries)})

        return reduced_summary
