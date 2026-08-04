"""
Scoring de la performance des AVS à partir de trois signaux déjà présents
dans la base : ponctualité des rapports (RapportJournalier.statutRemise),
ponctualité de présence (Presence.statut) et notes reçues des familles
(Appreciation.note). Pondération simple et documentée, à remplacer plus
tard par un modèle supervisé une fois qu'il y aura assez d'historique
pour entraîner quelque chose de plus fin (voir README.md).
"""

import logging

from app.schemas.performance import ReponsePerformanceAvs, RequetePerformanceAvs, ScoreAvs
from app.services import donnees
from app.training.registry import charger_derniere_version

logger = logging.getLogger("prm-spadcm-ia")

# Pondération du score global (somme = 1.0) — ajustable sans redéployer
# le reste du service si besoin d'un tuning rapide.
POIDS_PONCTUALITE_RAPPORTS = 0.4
POIDS_PONCTUALITE_PRESENCE = 0.35
POIDS_APPRECIATIONS = 0.25

# Modèle entraîné (voir app/training/entrainer_performance_avs.py), chargé une
# seule fois au démarrage du process. None tant qu'aucun entraînement n'a été
# lancé : `score_modele` reste alors absent de la réponse, sans erreur (voir
# README.md, principe de dégradation gracieuse).
try:
    _RESULTAT_MODELE = charger_derniere_version("performance_avs")
except Exception:  # noqa: BLE001
    logger.warning("Impossible de charger le modèle 'performance_avs' entraîné, on continue sans.", exc_info=True)
    _RESULTAT_MODELE = None
_MODELE_PERFORMANCE = _RESULTAT_MODELE[0] if _RESULTAT_MODELE else None


def _score_modele(taux_ponctualite_rapports: float, taux_presence_a_temps: float, nombre_rapports: int) -> float | None:
    if _MODELE_PERFORMANCE is None:
        return None
    try:
        import pandas as pd

        entree = pd.DataFrame([{
            "taux_ponctualite_rapports": taux_ponctualite_rapports,
            "taux_presence_a_temps": taux_presence_a_temps,
            "nombre_rapports": nombre_rapports,
        }])
        return round(float(_MODELE_PERFORMANCE.predict(entree)[0]), 2)
    except Exception:  # noqa: BLE001
        logger.warning("Échec de la prédiction du modèle 'performance_avs', on continue sans.", exc_info=True)
        return None


def _taux(nb_ok: int, nb_total: int) -> float:
    return round(100 * nb_ok / nb_total, 1) if nb_total else 100.0  # pas de données = pas de pénalité


async def _score_pour_avs(avs: dict, jours: int) -> ScoreAvs:
    avs_id = str(avs["_id"])
    rapports = await donnees.recuperer_rapports_avs(avs_id, jours)
    presences = await donnees.recuperer_presences_avs(avs_id, jours)
    appreciations = await donnees.recuperer_appreciations_avs(avs_id, jours)

    nb_rapports_a_temps = sum(1 for r in rapports if r.get("statutRemise") == "a_temps")
    nb_presences_a_temps = sum(1 for p in presences if p.get("statut") == "present")

    taux_ponctualite_rapports = _taux(nb_rapports_a_temps, len(rapports))
    taux_presence_a_temps = _taux(nb_presences_a_temps, len(presences))

    notes = [a["note"] for a in appreciations if a.get("note") is not None]
    note_moyenne = round(sum(notes) / len(notes), 2) if notes else None
    # Note sur 5 -> ramenée sur 100 pour la pondération ; si aucune note,
    # on neutralise ce facteur plutôt que de pénaliser un AVS peu noté.
    score_appreciations = (note_moyenne / 5 * 100) if note_moyenne is not None else 100.0

    score_global = round(
        taux_ponctualite_rapports * POIDS_PONCTUALITE_RAPPORTS
        + taux_presence_a_temps * POIDS_PONCTUALITE_PRESENCE
        + score_appreciations * POIDS_APPRECIATIONS,
        1,
    )

    return ScoreAvs(
        avs_id=avs_id,
        nom_complet=f"{avs.get('prenom', '')} {avs.get('nom', '')}".strip() or None,
        score_global=score_global,
        taux_ponctualite_rapports=taux_ponctualite_rapports,
        taux_presence_a_temps=taux_presence_a_temps,
        note_moyenne_appreciations=note_moyenne,
        nombre_rapports=len(rapports),
        nombre_presences=len(presences),
        nombre_appreciations=len(appreciations),
        score_modele=_score_modele(taux_ponctualite_rapports, taux_presence_a_temps, len(rapports)),
    )


async def calculer_performance(requete: RequetePerformanceAvs) -> ReponsePerformanceAvs:
    if requete.avs_id:
        avs = await donnees.recuperer_avs(requete.avs_id)
        liste_avs = [avs] if avs else []
    else:
        liste_avs = await donnees.lister_avs()

    resultats = [await _score_pour_avs(avs, requete.jours) for avs in liste_avs]
    resultats.sort(key=lambda s: s.score_global, reverse=True)

    return ReponsePerformanceAvs(periode_jours=requete.jours, resultats=resultats)
