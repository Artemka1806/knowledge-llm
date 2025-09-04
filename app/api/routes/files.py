from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile


router = APIRouter()


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    settings = request.app.state.settings
    # Save uploaded file into DATA_DIR
    if not file.filename:
        raise HTTPException(status_code=400, detail="Не задано ім'я файлу")
    dst_dir = settings.data_path
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / file.filename
    try:
        with open(dst_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка збереження файлу: {e}")
    return {"status": "Файл завантажено", "filename": file.filename, "path": str(dst_path)}


@router.get("/files")
def list_files(request: Request):
    base = request.app.state.settings.data_path
    base.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(base.iterdir() if base.exists() else [], key=lambda x: x.name.lower()):
        if p.is_file():
            stat = p.stat()
            items.append({
                "name": p.name,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
            })
    return {"files": items}

