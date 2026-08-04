"""
Extraction d'intention en langage naturel -> requête structurée.

Pourquoi pas du "tool calling" natif ? Parce que `llm.py` fait exprès
d'abstraire 4 fournisseurs très différents (Gemini/Groq/Ollama/
Anthropic) derrière un seul `generer_texte(system_prompt, messages)` en
texte brut, pour rester simple et permettre de changer de fournisseur
via une seule variable d'env (voir README.md). Refaire ça proprement en
tool calling natif obligerait à un adaptateur différent par fournisseur.
On obtient le même résultat en 2 appels texte :

  1. ce module : le LLM renvoie un petit JSON qui dit CE QUE l'utilisateur
     cherche (quel AVS, quel patient, quelle période) ;
  2. chat.py : on va chercher UNIQUEMENT ces données précises en base
     (avec vérification des droits, voir permissions.py), puis on
     rappelle le LLM pour rédiger la réponse finale avec ces données.

Aucune donnée n'est jamais devinée par ce module : il ne fait que lire
l'intention, jamais produire de contenu factuel lui-même.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel

from app.services.llm import generer_texte

logger = logging.getLogger("prm-spadcm-ia")

TypeIntention = Literal[
    "statut_avs",       # "est-ce que l'AVS X est à l'heure / en intervention ?"
    "statut_patient",   # "où en est l'intervention chez le patient Y ?"
    "graphique",        # "montre-moi/trace l'évolution/les stats de..."
    "general",          # question générale, pas de recherche de données ciblée
]


class IntentionDetectee(BaseModel):
    intention: TypeIntention = "general"
    nom_avs: str | None = None
    nom_patient: str | None = None
    periode_jours: int = 7


_PROMPT_EXTRACTION = (
    "Tu extrais l'intention d'un message envoyé à l'assistant IA d'une "
    "plateforme de soins à domicile (PRM SPAD Cameroun). Réponds "
    "UNIQUEMENT avec un objet JSON, sans texte autour, sans balises "
    "markdown, avec exactement ces clés :\n"
    '{"intention": "statut_avs" | "statut_patient" | "graphique" | "general", '
    '"nom_avs": string ou null, '
    '"nom_patient": string ou null, '
    '"periode_jours": entier (par défaut 7)}\n\n'
    "Règles :\n"
    "- \"statut_avs\" : la question porte sur un(e) AVS précis(e) "
    "(pointage, ponctualité, intervention en cours). Mets son nom "
    "(même partiel/entre guillemets) dans nom_avs.\n"
    "- \"statut_patient\" : la question porte sur un patient précis "
    "(visite en cours, dernier rapport). Mets son nom dans nom_patient.\n"
    "- \"graphique\" : la personne demande une visualisation, des "
    "statistiques, une évolution, un classement, une tendance.\n"
    "- \"general\" : toute autre question (fonctionnement de l'appli, "
    "question médicale générale, etc.) — nom_avs et nom_patient à null.\n"
    "- N'invente jamais un nom qui n'est pas dans le message."
)


def _extraire_bloc_json(texte: str) -> str:
    """Le LLM respecte rarement 100% du temps la consigne \"que du JSON\" :
    on isole le premier bloc { ... } plutôt que de faire échouer toute la
    requête sur un simple caractère de trop avant/après."""
    debut = texte.find("{")
    fin = texte.rfind("}")
    if debut == -1 or fin == -1 or fin < debut:
        raise ValueError("Aucun JSON trouvé dans la réponse du LLM.")
    return texte[debut : fin + 1]


async def extraire_intention(message: str) -> IntentionDetectee:
    try:
        brut = await generer_texte(
            system_prompt=_PROMPT_EXTRACTION,
            messages=[{"role": "user", "content": message}],
            max_tokens=200,
        )
        donnees_json = json.loads(_extraire_bloc_json(brut))
        return IntentionDetectee(**donnees_json)
    except Exception:  # noqa: BLE001
        # Dégradation gracieuse : en cas d'échec (JSON mal formé, LLM
        # indisponible pour cet appel), on retombe sur "general" — le chat
        # continue de fonctionner comme avant (RAG patient simple), on ne
        # bloque jamais la conversation pour un problème d'extraction.
        logger.warning("Échec de l'extraction d'intention, on retombe sur 'general'.", exc_info=True)
        return IntentionDetectee()
