from llama_index.core import Settings, PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.gemini import GeminiEmbedding

from .config import AppSettings, RuntimeConfig


def configure_llm(settings: AppSettings, runtime: RuntimeConfig) -> None:
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY не задано у .env")

    Settings.llm = GoogleGenAI(model=runtime.llm_model, api_key=settings.GOOGLE_API_KEY)
    Settings.embed_model = GeminiEmbedding(model=runtime.embed_model, api_key=settings.GOOGLE_API_KEY)

    apply_text_qa_template(runtime.system_prompt)


def apply_text_qa_template(system_prompt: str) -> None:
    Settings.text_qa_template = PromptTemplate(
        (
            f"{system_prompt}\n\n"
            "Тобі може бути надано контекст з документів.\n\n"
            "Контекст (може бути порожнім або нерелевантним):\n{context_str}\n\n"
            "Запит користувача: {query_str}\n\n"
            "Інструкції:\n"
            "- Якщо контекст має відношення до запиту — використай його.\n"
            "- Якщо контексту бракує або він нерелевантний — відповідай за загальними знаннями.\n"
            "- Відповідай українською, стисло та по суті."
        )
    )
