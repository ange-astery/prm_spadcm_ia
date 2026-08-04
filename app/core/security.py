"""
Ce service n'est JAMAIS appelé directement par les apps Flutter : seul le
backend Node (prm_spadcm_backend) l'appelle, côté serveur, après avoir
lui-même authentifié l'utilisateur (JWT) et vérifié son rôle. La seule
protection nécessaire ici est donc de vérifier que l'appelant est bien
le backend Node, via un secret partagé — pas un système d'auth complet.

Voir README.md, section "Communication avec le backend Node" pour le
schéma d'ensemble.
"""

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def verifier_token_service(x_internal_token: str = Header(default="")) -> None:
    settings = get_settings()

    if not settings.ia_service_token:
        # Pas de secret configuré : on refuse tout par défaut plutôt que de
        # tourner "ouvert" par accident (voir .env.example).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IA_SERVICE_TOKEN non configuré côté service IA.",
        )

    if x_internal_token != settings.ia_service_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de service invalide ou manquant (en-tête X-Internal-Token).",
        )
