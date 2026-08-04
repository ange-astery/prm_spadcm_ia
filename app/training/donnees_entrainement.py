"""
Extraction des données d'entraînement depuis MongoDB (lecture seule,
connexion SYNCHRONE via pymongo — les scripts d'entraînement ne tournent
pas dans le service FastAPI, donc pas besoin de motor/async ici, voir
README.md section 11.5.2).

Chaque fonction de génération de dataset a un mode `synthetique=True` :
utile tant que le projet n'a pas encore assez d'historique réel pour
entraîner quoi que ce soit d'utile (quelques dizaines de rapports ne
suffisent pas). Les données synthétiques restent physiologiquement
plausibles (mêmes bornes que les seuils de app/services/alertes.py) afin
que le pipeline (et ses courbes de performance) soit vérifiable de bout
en bout dès aujourd'hui, puis ré-exécuté sur données réelles au fur et à
mesure qu'elles s'accumulent (voir README.md 11.5.7).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymongo import MongoClient

from app.core.config import get_settings

SEUIL_MIN_EXEMPLES_REELS = 60  # en-dessous, les courbes n'ont pas de sens statistique


@dataclass
class JeuDeDonnees:
    X: pd.DataFrame
    y: pd.Series
    source: str  # "reel" | "synthetique"
    colonnes: list[str]


def _client_synchrone() -> MongoClient:
    settings = get_settings()
    return MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)


# --------------------------------------------------------------------------
# Dataset 1 : anomalies de constantes vitales (classification binaire)
# --------------------------------------------------------------------------

CHAMPS_VITAUX = ["pouls", "temperature", "spo2", "glycemie", "frequenceRespiratoire"]


def _extraire_relevés_vitaux_reels() -> pd.DataFrame:
    settings = get_settings()
    client = _client_synchrone()
    db = client[settings.mongo_db_name]

    lignes = []
    curseur = db["rapportjournaliers"].find({}, {"parametresVitaux": 1}).limit(5000)
    for doc in curseur:
        for releve in doc.get("parametresVitaux", []) or []:
            ligne = {champ: releve.get(champ) for champ in CHAMPS_VITAUX}
            lignes.append(ligne)
    client.close()
    return pd.DataFrame(lignes)


def _generer_releves_synthetiques(n: int = 600, taux_anomalie: float = 0.18) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_anormaux = int(n * taux_anomalie)
    n_normaux = n - n_anormaux

    normaux = pd.DataFrame({
        "pouls": rng.normal(75, 8, n_normaux).clip(55, 100),
        "temperature": rng.normal(36.8, 0.3, n_normaux).clip(36.0, 37.5),
        "spo2": rng.normal(97, 1.2, n_normaux).clip(94, 100),
        "glycemie": rng.normal(1.0, 0.15, n_normaux).clip(0.7, 1.4),
        "frequenceRespiratoire": rng.normal(16, 2, n_normaux).clip(12, 20),
    })

    # Anomalies : on tire, pour chaque exemple, 1 à 2 constantes qui sortent
    # franchement des seuils de app/services/alertes.py.
    anormaux = pd.DataFrame({
        "pouls": rng.choice([rng.normal(35, 8), rng.normal(140, 15)], n_anormaux) if n_anormaux else [],
        "temperature": rng.normal(36.8, 0.3, n_anormaux),
        "spo2": rng.normal(97, 1.2, n_anormaux),
        "glycemie": rng.normal(1.0, 0.15, n_anormaux),
        "frequenceRespiratoire": rng.normal(16, 2, n_anormaux),
    }) if n_anormaux else pd.DataFrame(columns=CHAMPS_VITAUX)

    # Ré-écrase aléatoirement une des colonnes de chaque ligne "anormale"
    # avec une valeur hors seuil, pour varier le type d'anomalie simulée.
    for i in range(len(anormaux)):
        champ = random.choice(["pouls", "temperature", "spo2", "glycemie"])
        if champ == "pouls":
            anormaux.iat[i, anormaux.columns.get_loc(champ)] = float(rng.choice([rng.uniform(25, 45), rng.uniform(125, 160)]))
        elif champ == "temperature":
            anormaux.iat[i, anormaux.columns.get_loc(champ)] = float(rng.choice([rng.uniform(33, 34.8), rng.uniform(38.7, 40.5)]))
        elif champ == "spo2":
            anormaux.iat[i, anormaux.columns.get_loc(champ)] = float(rng.uniform(78, 91))
        elif champ == "glycemie":
            anormaux.iat[i, anormaux.columns.get_loc(champ)] = float(rng.choice([rng.uniform(0.2, 0.55), rng.uniform(2.2, 3.5)]))

    normaux["anomalie"] = 0
    if len(anormaux):
        anormaux["anomalie"] = 1
    donnees = pd.concat([normaux, anormaux], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return donnees


def construire_dataset_anomalies_vitaux(forcer_synthetique: bool = False) -> JeuDeDonnees:
    if not forcer_synthetique:
        reel = _extraire_relevés_vitaux_reels()
        reel = reel.dropna(thresh=2)  # au moins 2 constantes renseignées
        if len(reel) >= SEUIL_MIN_EXEMPLES_REELS:
            # Étiquetage faible (weak supervision) : on réutilise les
            # mêmes seuils que le service par règles pour poser une
            # première vérité terrain, le temps qu'un vrai retour humain
            # (alerte confirmée/rejetée) alimente un futur ré-entraînement
            # plus fiable — voir README.md 11.5.7.
            from app.services.alertes import SEUILS

            def est_anormal(ligne) -> int:
                for champ, seuils in SEUILS.items():
                    val = ligne.get(champ)
                    if val is None:
                        continue
                    if seuils["bas"] is not None and val < seuils["bas"]:
                        return 1
                    if seuils["haut"] is not None and val > seuils["haut"]:
                        return 1
                return 0

            reel["anomalie"] = reel.apply(est_anormal, axis=1)
            reel = reel.fillna(reel.median(numeric_only=True))
            return JeuDeDonnees(
                X=reel[CHAMPS_VITAUX], y=reel["anomalie"], source="reel", colonnes=CHAMPS_VITAUX
            )

    synthetique = _generer_releves_synthetiques()
    return JeuDeDonnees(
        X=synthetique[CHAMPS_VITAUX], y=synthetique["anomalie"], source="synthetique", colonnes=CHAMPS_VITAUX
    )


# --------------------------------------------------------------------------
# Dataset 2 : score de performance AVS (régression)
# --------------------------------------------------------------------------

CHAMPS_PERFORMANCE = [
    "taux_ponctualite_rapports",
    "taux_presence_a_temps",
    "note_moyenne_appreciations",
    "nombre_rapports",
]


def _extraire_performance_avs_reelle() -> pd.DataFrame:
    """Reconstruit, par AVS, les mêmes agrégats que
    app/services/performance_avs.py (calcul par règles) pour servir de
    features ; la cible est la note moyenne des appréciations (seul
    signal réellement fourni par un tiers humain, donc le plus proche
    d'une vérité terrain disponible aujourd'hui)."""
    settings = get_settings()
    client = _client_synchrone()
    db = client[settings.mongo_db_name]

    lignes = []
    for avs in db["utilisateurs"].find({"role": "avs"}, {"_id": 1}):
        avs_id = avs["_id"]
        rapports = list(db["rapportjournaliers"].find({"avsId": avs_id}, {"statutRemise": 1}))
        presences = list(db["presences"].find({"avsId": avs_id}, {"statut": 1}))
        appreciations = list(db["appreciations"].find({"avsId": avs_id}, {"note": 1}))

        if not rapports or not appreciations:
            continue

        taux_rapports = 100 * sum(1 for r in rapports if r.get("statutRemise") == "a_temps") / len(rapports)
        taux_presence = 100 * sum(1 for p in presences if p.get("statut") == "present") / len(presences) if presences else 100.0
        notes = [a["note"] for a in appreciations if a.get("note") is not None]
        note_moyenne = sum(notes) / len(notes) if notes else None
        if note_moyenne is None:
            continue

        lignes.append({
            "taux_ponctualite_rapports": taux_rapports,
            "taux_presence_a_temps": taux_presence,
            "note_moyenne_appreciations": note_moyenne,
            "nombre_rapports": len(rapports),
        })
    client.close()
    return pd.DataFrame(lignes)


def _generer_performance_synthetique(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    taux_rapports = rng.uniform(40, 100, n)
    taux_presence = rng.uniform(50, 100, n)
    nombre_rapports = rng.integers(5, 120, n)

    # La note (cible) suit une combinaison bruitée des deux taux, pour
    # simuler une corrélation réaliste (mais imparfaite) entre ponctualité
    # et satisfaction perçue par la famille.
    bruit = rng.normal(0, 0.35, n)
    note = 1 + 4 * (0.55 * taux_rapports / 100 + 0.45 * taux_presence / 100) + bruit
    note = np.clip(note, 1, 5)

    return pd.DataFrame({
        "taux_ponctualite_rapports": taux_rapports,
        "taux_presence_a_temps": taux_presence,
        "note_moyenne_appreciations": note,  # utilisée seulement pour compat colonnes, retirée des features
        "nombre_rapports": nombre_rapports,
        "cible_note": note,
    })


def construire_dataset_performance_avs(forcer_synthetique: bool = False) -> JeuDeDonnees:
    if not forcer_synthetique:
        reel = _extraire_performance_avs_reelle()
        if len(reel) >= SEUIL_MIN_EXEMPLES_REELS:
            features = ["taux_ponctualite_rapports", "taux_presence_a_temps", "nombre_rapports"]
            return JeuDeDonnees(
                X=reel[features], y=reel["note_moyenne_appreciations"], source="reel", colonnes=features
            )

    synthetique = _generer_performance_synthetique()
    features = ["taux_ponctualite_rapports", "taux_presence_a_temps", "nombre_rapports"]
    return JeuDeDonnees(X=synthetique[features], y=synthetique["cible_note"], source="synthetique", colonnes=features)
