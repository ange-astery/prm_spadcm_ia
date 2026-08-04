"""
Ce service lit MongoDB en lecture seule mais n'y écrit jamais directement
(voir app/db/mongo.py). Quand un résultat doit être persisté (ex: remplir
RapportJournalier.resumeIA), il est transmis au backend Node via un appel
HTTP, protégé par un second secret partagé (BACKEND_CALLBACK_TOKEN),
distinct de IA_SERVICE_TOKEN (qui protège le sens Node -> IA).

Voir README.md, section "Endpoint à ajouter côté backend Node" pour le
code de la route Express correspondante (PATCH /api/rapports/:id/resume-ia).
"""

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger("prm-spadcm-ia")


async def enregistrer_resume_rapport(rapport_id: str, resume: str) -> bool:
    settings = get_settings()

    if not settings.backend_callback_token:
        logger.warning(
            "BACKEND_CALLBACK_TOKEN non configuré : résumé généré mais non "
            "persisté côté backend Node (rapport_id=%s)",
            rapport_id,
        )
        return False

    url = f"{settings.backend_url.rstrip('/')}/api/rapports/{rapport_id}/resume-ia"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            reponse = await client.patch(
                url,
                headers={"X-Internal-Token": settings.backend_callback_token},
                json={"resumeIA": resume},
            )
        return reponse.status_code < 300
    except httpx.HTTPError:
        logger.exception("Échec du callback vers le backend Node (rapport_id=%s)", rapport_id)
        return False
