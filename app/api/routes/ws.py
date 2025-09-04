import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor

from app.core.config import RuntimeConfig
from app.services.rag import build_prompt_from_history, should_fallback_to_llm


router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "ready"})

    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
            except Exception:
                await websocket.send_json({"type": "error", "error": "Невірний формат повідомлення. Надсилайте JSON."})
                continue

            if data.get("type") != "ask":
                await websocket.send_json({"type": "error", "error": "Підтримується лише type=ask."})
                continue

            app_obj = getattr(websocket, "app", None) or websocket.scope.get("app")
            index = app_obj.state.index
            runtime: RuntimeConfig = app_obj.state.runtime
            if index is None:
                await websocket.send_json({"type": "error", "error": "Індекс ще не готовий"})
                continue

            message = (data.get("message") or "").strip()
            history = data.get("history") or []
            if not message:
                await websocket.send_json({"type": "error", "error": "Порожнє питання"})
                continue

            prompt = build_prompt_from_history(runtime.system_prompt, history, message)

            try:
                final_text = ""
                if await should_fallback_to_llm(message, index, runtime):
                    llm = Settings.llm
                    acomplete = getattr(llm, "acomplete", None)
                    if callable(acomplete):
                        comp = await acomplete(f"{runtime.system_prompt}\nПитання: {prompt}")
                    else:
                        # Offload sync completion to a thread
                        import asyncio as _asyncio
                        comp = await _asyncio.to_thread(llm.complete, f"{runtime.system_prompt}\nПитання: {prompt}")
                    final_text = str(comp)
                    await websocket.send_json({"type": "token", "token": final_text})
                else:
                    aengine = index.as_query_engine(
                        streaming=True,
                        similarity_top_k=runtime.rag_top_k,
                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=runtime.rag_cutoff)],
                    )
                    resp = await aengine.aquery(message)
                    final_text_tokens = []
                    async_gen_attr = getattr(resp, "async_response_gen", None)
                    if async_gen_attr is not None:
                        agen = async_gen_attr() if callable(async_gen_attr) else async_gen_attr
                        try:
                            async for token in agen:
                                final_text_tokens.append(token)
                                await websocket.send_json({"type": "token", "token": token})
                        except TypeError:
                            pass
                    if not final_text_tokens:
                        token_stream = getattr(resp, "response_gen", None)
                        if token_stream is not None:
                            for token in token_stream:
                                final_text_tokens.append(token)
                                await websocket.send_json({"type": "token", "token": token})
                    if final_text_tokens:
                        final_text = "".join(final_text_tokens)
                    else:
                        final_text = str(resp)
                        await websocket.send_json({"type": "token", "token": final_text})
            except Exception as e:
                await websocket.send_json({"type": "error", "error": str(e)})
                continue

            new_history = list(history) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": final_text},
            ]
            await websocket.send_json({"type": "done", "message": final_text, "history": new_history})
    except WebSocketDisconnect:
        return
