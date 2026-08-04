"""
Registre simple de versions de modèles, en local sur disque :

    app/models/<nom_modele>/v1/model.joblib
    app/models/<nom_modele>/v1/metrics.json
    app/models/<nom_modele>/v1/courbes/*.png
    app/models/<nom_modele>/dernier.txt          <- contient juste "v1"

Volontairement sans base de données ni service externe (MLflow, etc.) :
le projet n'en est pas encore là (voir README.md, section 11.5.7 "boucle
d'amélioration continue"). `dernier.txt` fait office de pointeur "modèle
en production" que les services d'inférence (performance_avs.py,
alertes.py) consultent pour savoir s'il existe un modèle entraîné à
utiliser, sinon ils retombent sur leur logique à base de règles.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

DOSSIER_MODELES = Path(__file__).resolve().parent.parent / "models"


def _dossier_modele(nom_modele: str) -> Path:
    return DOSSIER_MODELES / nom_modele


def _prochaine_version(nom_modele: str) -> str:
    dossier = _dossier_modele(nom_modele)
    if not dossier.exists():
        return "v1"
    existantes = [p.name for p in dossier.iterdir() if p.is_dir() and p.name.startswith("v")]
    numeros = [int(n[1:]) for n in existantes if n[1:].isdigit()]
    return f"v{max(numeros, default=0) + 1}"


def sauvegarder_version(
    nom_modele: str,
    modele: Any,
    metriques: dict,
    courbes: list[str],
    meta: dict | None = None,
) -> str:
    """Sauvegarde une nouvelle version versionnée du modèle + son rapport
    d'étude de performance, et met à jour le pointeur "dernier". Renvoie
    l'identifiant de version créé (ex. "v3")."""
    version = _prochaine_version(nom_modele)
    dossier_version = _dossier_modele(nom_modele) / version
    dossier_version.mkdir(parents=True, exist_ok=True)

    joblib.dump(modele, dossier_version / "model.joblib")

    rapport = {
        "modele": nom_modele,
        "version": version,
        "date_entrainement": datetime.now(timezone.utc).isoformat(),
        "metriques": metriques,
        "courbes": [str(Path(c).relative_to(dossier_version)) for c in courbes],
        "meta": meta or {},
    }
    with open(dossier_version / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    with open(_dossier_modele(nom_modele) / "dernier.txt", "w", encoding="utf-8") as f:
        f.write(version)

    return version


def charger_derniere_version(nom_modele: str) -> tuple[Any, dict] | None:
    """Renvoie (modele_charge, metriques_dict) pour la dernière version
    entraînée, ou None si aucun modèle n'a encore été entraîné pour ce
    nom — c'est ce None que les services d'inférence utilisent pour
    savoir s'ils doivent retomber sur leur logique à base de règles."""
    pointeur = _dossier_modele(nom_modele) / "dernier.txt"
    if not pointeur.exists():
        return None

    version = pointeur.read_text(encoding="utf-8").strip()
    dossier_version = _dossier_modele(nom_modele) / version
    chemin_modele = dossier_version / "model.joblib"
    chemin_metriques = dossier_version / "metrics.json"
    if not chemin_modele.exists():
        return None

    modele = joblib.load(chemin_modele)
    metriques = {}
    if chemin_metriques.exists():
        with open(chemin_metriques, encoding="utf-8") as f:
            metriques = json.load(f)
    return modele, metriques


def lister_versions(nom_modele: str) -> list[dict]:
    dossier = _dossier_modele(nom_modele)
    if not dossier.exists():
        return []
    rapports = []
    for sous_dossier in sorted(dossier.iterdir()):
        chemin_metriques = sous_dossier / "metrics.json"
        if chemin_metriques.exists():
            with open(chemin_metriques, encoding="utf-8") as f:
                rapports.append(json.load(f))
    return rapports
