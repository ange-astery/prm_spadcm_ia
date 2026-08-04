"""
Détection d'anomalies dans le dernier relevé de constantes d'un patient.
Basée sur des seuils de référence usuels (adulte/personne âgée) — PAS un
diagnostic, un simple signal pour attirer l'attention d'un professionnel.
Volontairement basé sur des règles simples et explicables comme premier
jet ; à faire évoluer vers un modèle statistique (ex. détection d'écarts
inhabituels PAR RAPPORT à l'historique propre du patient, plus pertinent
qu'un seuil générique) une fois qu'il y a assez d'historique par patient.

Ce service NE CRÉE JAMAIS d'Alerte directement (voir README.md, principe
"le service IA ne décide jamais seul") : il renvoie une proposition que
le backend Node (ou un humain) décide de transformer en Alerte réelle.
"""

import logging

from app.schemas.performance import (
    AnomalieDetectee,
    ReponseAlertesIntelligentes,
    RequeteAlertesIntelligentes,
)
from app.services import donnees
from app.services.evolution import extraire_nombre  # réutilisé tel quel
from app.training.registry import charger_derniere_version

logger = logging.getLogger("prm-spadcm-ia")

# Modèle entraîné (voir app/training/entrainer_anomalies_vitaux.py). En son
# absence, on continue avec les seuils fixes ci-dessous — aucune régression
# de comportement pour un déploiement qui n'a jamais lancé d'entraînement.
try:
    _RESULTAT_MODELE = charger_derniere_version("anomalies_vitaux")
except Exception:  # noqa: BLE001
    logger.warning("Impossible de charger le modèle 'anomalies_vitaux' entraîné, on continue avec les seuils fixes.", exc_info=True)
    _RESULTAT_MODELE = None
_MODELE_ANOMALIES = _RESULTAT_MODELE[0] if _RESULTAT_MODELE else None
_SEUIL_CONFIANCE_MODELE = 0.75  # ne propose une anomalie "modèle" que si le classifieur est raisonnablement confiant

# Seuils indicatifs, à ajuster avec un médecin référent du projet.
SEUILS = {
    "temperature": {"bas": 35.0, "haut": 38.5},
    "spo2": {"bas": 92.0, "haut": None},
    "pouls": {"bas": 50.0, "haut": 120.0},
    "glycemie": {"bas": 0.6, "haut": 2.0},  # g/L, ordre de grandeur
}


def _evaluer_releve(releve: dict) -> list[AnomalieDetectee]:
    anomalies: list[AnomalieDetectee] = []

    for champ, seuils in SEUILS.items():
        valeur = extraire_nombre(releve.get(champ))
        if valeur is None:
            continue

        if seuils["bas"] is not None and valeur < seuils["bas"]:
            anomalies.append(
                AnomalieDetectee(
                    champ=champ,
                    valeur=str(valeur),
                    seuil_reference=f"< {seuils['bas']}",
                    gravite="urgent" if champ in ("spo2", "temperature") else "attention",
                    explication=f"{champ} anormalement bas ({valeur}), en-dessous du seuil de référence {seuils['bas']}.",
                )
            )
        if seuils["haut"] is not None and valeur > seuils["haut"]:
            anomalies.append(
                AnomalieDetectee(
                    champ=champ,
                    valeur=str(valeur),
                    seuil_reference=f"> {seuils['haut']}",
                    gravite="urgent" if champ == "temperature" else "attention",
                    explication=f"{champ} anormalement élevé ({valeur}), au-dessus du seuil de référence {seuils['haut']}.",
                )
            )

    if releve.get("detresseRespiratoire") is True:
        anomalies.append(
            AnomalieDetectee(
                champ="detresseRespiratoire",
                valeur="true",
                seuil_reference="false attendu",
                gravite="urgent",
                explication="Détresse respiratoire signalée dans l'évaluation des besoins.",
            )
        )

    return anomalies


def _evaluer_avec_modele(releve: dict) -> AnomalieDetectee | None:
    """Second avis, complémentaire aux seuils fixes : un classifieur
    entraîné (voir app/training/entrainer_anomalies_vitaux.py) peut
    repérer une combinaison de constantes individuellement dans les
    clous mais conjointement inhabituelle. Ne s'exécute que si un modèle
    a été entraîné (sinon _MODELE_ANOMALIES est None, voir plus haut)."""
    if _MODELE_ANOMALIES is None:
        return None

    from app.training.donnees_entrainement import CHAMPS_VITAUX

    valeurs = {champ: extraire_nombre(releve.get(champ)) for champ in CHAMPS_VITAUX}
    if sum(v is not None for v in valeurs.values()) < 2:
        return None  # trop peu de constantes renseignées pour une prédiction utile

    try:
        import pandas as pd

        entree = pd.DataFrame([valeurs])
        entree = entree.fillna(entree.median(numeric_only=True)).fillna(0)
        probabilite_anomalie = float(_MODELE_ANOMALIES.predict_proba(entree)[0][1])
    except Exception:  # noqa: BLE001
        logger.warning("Échec de la prédiction du modèle 'anomalies_vitaux', on ignore ce second avis.", exc_info=True)
        return None

    if probabilite_anomalie < _SEUIL_CONFIANCE_MODELE:
        return None

    return AnomalieDetectee(
        champ="combinaison_constantes",
        valeur=", ".join(f"{k}={v}" for k, v in valeurs.items() if v is not None),
        seuil_reference=f"confiance modèle ≥ {_SEUIL_CONFIANCE_MODELE:.0%}",
        gravite="attention",
        explication=(
            f"Le modèle entraîné juge cette combinaison de constantes inhabituelle "
            f"(confiance {probabilite_anomalie:.0%}), sans qu'aucune ne dépasse individuellement "
            "un seuil fixe — à vérifier par un professionnel."
        ),
        source="modele",
    )


async def detecter_anomalies(requete: RequeteAlertesIntelligentes) -> ReponseAlertesIntelligentes:
    rapports = await donnees.recuperer_rapports_recents(requete.patient_id, jours=1)

    anomalies: list[AnomalieDetectee] = []
    for rapport in rapports:
        for releve in rapport.get("parametresVitaux", []) or []:
            anomalies.extend(_evaluer_releve(releve))
            anomalie_modele = _evaluer_avec_modele(releve)
            if anomalie_modele is not None:
                anomalies.append(anomalie_modele)

        besoins = (rapport.get("evaluationBesoins") or {}).get("respirer") or {}
        if besoins.get("detresseRespiratoire"):
            anomalies.append(
                AnomalieDetectee(
                    champ="detresseRespiratoire",
                    valeur="true",
                    seuil_reference="false attendu",
                    gravite="urgent",
                    explication="Détresse respiratoire signalée dans l'évaluation des besoins du rapport.",
                )
            )

    gravite_max = max((a.gravite for a in anomalies), default=None, key=lambda g: {"info": 0, "attention": 1, "urgent": 2}[g])

    return ReponseAlertesIntelligentes(
        patient_id=requete.patient_id,
        anomalies=anomalies,
        proposition_alerte=gravite_max == "urgent",
        type_alerte_propose="medicale" if gravite_max == "urgent" else None,
    )
