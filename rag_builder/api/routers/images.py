"""Service des images extraites des documents (référencées par rag-image://)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from rag_builder.api.deps import current_user, get_service
from rag_builder.core.rag_service import RagService
from rag_builder.db.models import User

router = APIRouter(prefix="/collections/{name}/images", tags=["images"])

_MEDIA = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


@router.get("/{path:path}")
def serve_image(
    name: str,
    path: str,
    _user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
):
    # La référence interne est <collection>/<doc_id>/<fichier> ; on la reconstruit.
    resolved = svc.image_store.resolve(f"{name}/{path}")
    if resolved is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image introuvable")
    media = _MEDIA.get(resolved.suffix.lower(), "application/octet-stream")
    return FileResponse(resolved, media_type=media)
