from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])

_CLES_PAR_FOURNISSEUR = {
    "anthropic": "anthropic_api_key",
    "gemini": "gemini_api_key",
    "groq": "groq_api_key",
    "ollama": None,  # pas de clé : juste une URL locale, toujours "configuré"
}


@router.get("/health")
async def health():
    settings = get_settings()
    cle_attendue = _CLES_PAR_FOURNISSEUR.get(settings.llm_provider.lower().strip())
    llm_configure = cle_attendue is None or bool(getattr(settings, cle_attendue, ""))

    return {
        "status": "ok",
        "env": settings.env,
        "llm_provider": settings.llm_provider,
        "llm_configure": llm_configure,
        "mongo_configure": bool(settings.mongo_uri),
    }
