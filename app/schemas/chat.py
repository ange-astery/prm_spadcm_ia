from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import ContexteUtilisateur


class MessageHistorique(BaseModel):
    role: Literal["user", "assistant"]
    contenu: str


class RequeteChat(BaseModel):
    message: str = Field(..., min_length=1)
    historique: list[MessageHistorique] = Field(default_factory=list)
    utilisateur: ContexteUtilisateur
    # Optionnel : si le message concerne un patient précis (ex. AVS/médecin
    # qui consultent un dossier), permet d'enrichir le contexte RAG avec les
    # données de CE patient. Absent pour un patient/famille (son propre
    # dossier est retrouvé automatiquement via utilisateur.id).
    patient_id: str | None = None


class ReponseChat(BaseModel):
    reponse: str
    sources: list[str] = Field(
        default_factory=list,
        description="Description courte des documents Mongo utilisés pour construire la réponse (traçabilité).",
    )
    # Présent uniquement quand l'intention détectée est "graphique" (voir
    # services/intention.py) et qu'assez de données existent pour en tracer
    # un. Format générique consommé tel quel par fl_chart côté Flutter (voir
    # services/graphiques.py) — jamais de chiffres "inventés" par le LLM ici.
    donnees_graphique: dict | None = Field(
        default=None,
        description="Données prêtes à tracer : {type, titre, labels, series:[{nom, valeurs}]}.",
    )
