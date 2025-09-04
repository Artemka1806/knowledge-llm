from fastapi import APIRouter, HTTPException, Request
from llama_index.core import Settings

from app.core.config import RuntimeConfig


router = APIRouter()


@router.get("/suggest")
def suggest_question(request: Request):
    runtime: RuntimeConfig = request.app.state.runtime
    try:
        prompt = (
            f"{runtime.system_prompt}\n\n"
            "Запропонуй одне прикладне, максимально корисне запитання до цього асистента. "
            "Відповідай лише самим запитанням, одним реченням, українською."
        )
        resp = Settings.llm.complete(prompt)
        text = getattr(resp, "text", None)
        text = (text if isinstance(text, str) else str(text or "")).strip()
        # Remove wrapping quotes if present
        if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
            text = text[1:-1].strip()
        if not text:
            raise ValueError("empty suggestion")
        return {"suggestion": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не вдалося згенерувати підказку: {e}")

