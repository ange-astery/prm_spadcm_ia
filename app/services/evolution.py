"""
Analyse de l'évolution de l'état de santé d'un patient à partir de
l'historique de ses rapports journaliers.

Volontairement SANS appel LLM : c'est un calcul statistique simple
(moyennes journalières + régression linéaire pour la tendance), donc
rapide, gratuit, reproductible, et explicable — un bon premier "modèle
maison" avant d'envisager quelque chose de plus sophistiqué une fois
qu'il y aura assez d'historique réel dans la base.
"""

import re
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from app.schemas.rapports import PointEvolution, ReponseEvolution, RequeteEvolution
from app.services import donnees

_NOMBRE_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def extraire_nombre(valeur) -> float | None:
    """Les constantes sont saisies en String côté Node (ex: "36.7", "98%").
    On extrait le premier nombre trouvé, sans lever d'exception sur les
    valeurs vides/mal formées (saisie terrain, pas toujours propre)."""
    if valeur is None:
        return None
    match = _NOMBRE_RE.search(str(valeur))
    return float(match.group().replace(",", ".")) if match else None


def _moyennes_par_jour(rapports: list[dict]) -> dict[date, dict[str, list[float]]]:
    par_jour: dict[date, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for rapport in rapports:
        date_rapport = rapport.get("date")
        jour = date_rapport.date() if hasattr(date_rapport, "date") else None
        if jour is None:
            continue

        for releve in rapport.get("parametresVitaux", []) or []:
            pouls = extraire_nombre(releve.get("pouls"))
            temperature = extraire_nombre(releve.get("temperature"))
            spo2 = extraire_nombre(releve.get("spo2"))

            if pouls is not None:
                par_jour[jour]["pouls"].append(pouls)
            if temperature is not None:
                par_jour[jour]["temperature"].append(temperature)
            if spo2 is not None:
                par_jour[jour]["spo2"].append(spo2)

    return par_jour


def _tendance(valeurs: list[float]) -> float:
    """Pente d'une régression linéaire simple (indice du jour -> valeur).
    Renvoyée telle quelle : positive = valeur en hausse dans le temps."""
    if len(valeurs) < 2:
        return 0.0
    x = np.arange(len(valeurs))
    y = np.array(valeurs)
    pente, _ = np.polyfit(x, y, 1)
    return float(pente)


async def analyser_evolution(requete: RequeteEvolution) -> ReponseEvolution:
    date_debut = date.today() - timedelta(days=requete.jours)
    rapports = await donnees.recuperer_rapports_patient(requete.patient_id, date_debut=date_debut)

    par_jour = _moyennes_par_jour(rapports)
    jours_tries = sorted(par_jour.keys())

    points = [
        PointEvolution(
            date=jour,
            pouls_moyen=round(sum(par_jour[jour]["pouls"]) / len(par_jour[jour]["pouls"]), 1)
            if par_jour[jour]["pouls"]
            else None,
            temperature_moyenne=round(sum(par_jour[jour]["temperature"]) / len(par_jour[jour]["temperature"]), 1)
            if par_jour[jour]["temperature"]
            else None,
            spo2_moyen=round(sum(par_jour[jour]["spo2"]) / len(par_jour[jour]["spo2"]), 1)
            if par_jour[jour]["spo2"]
            else None,
        )
        for jour in jours_tries
    ]

    if not points:
        return ReponseEvolution(
            patient_id=requete.patient_id,
            points=[],
            tendance="donnees_insuffisantes",
            analyse="Pas assez de relevés de constantes sur cette période pour dégager une tendance.",
        )

    temperatures = [p.temperature_moyenne for p in points if p.temperature_moyenne is not None]
    spo2s = [p.spo2_moyen for p in points if p.spo2_moyen is not None]

    pente_temp = _tendance(temperatures)
    pente_spo2 = _tendance(spo2s)

    # Règles simples, explicables : une base de départ, à affiner avec de
    # vraies données (voir README.md, "faire évoluer ce modèle").
    alerte_temp = pente_temp > 0.05  # température qui grimpe régulièrement
    alerte_spo2 = pente_spo2 < -0.3  # SpO2 qui baisse régulièrement

    if alerte_temp or alerte_spo2:
        tendance = "degradation_a_surveiller"
    elif abs(pente_temp) < 0.02 and abs(pente_spo2) < 0.1:
        tendance = "stable"
    else:
        tendance = "amelioration"

    phrases = [f"Analyse basée sur {len(rapports)} rapport(s) sur {requete.jours} jours."]
    if alerte_temp:
        phrases.append("La température moyenne présente une tendance à la hausse à surveiller.")
    if alerte_spo2:
        phrases.append("La saturation en oxygène (SpO2) présente une tendance à la baisse à surveiller.")
    if not alerte_temp and not alerte_spo2:
        phrases.append("Aucune tendance préoccupante détectée sur les constantes suivies.")

    return ReponseEvolution(
        patient_id=requete.patient_id,
        points=points,
        tendance=tendance,
        analyse=" ".join(phrases),
    )
